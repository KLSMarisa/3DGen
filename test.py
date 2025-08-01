import os
from omegaconf import OmegaConf
import warnings
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from trainer.trainer import *
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from diffusers import DPMSolverMultistepScheduler


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='main', help='experiment identifier')
    parser.add_argument('--savedir', type=str, default='./logs', help='path to save checkpoints and logs')
    parser.add_argument('--exp', type=str, default='diffusion', choices=['diffusion'], help='experiment type to run')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='experiment mode to run')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    """ Args about Data """
    parser.add_argument('--dataset', type=str, default='webvid')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--center_crop', default=True, action="store_true", help=('Whether to center crop the input images to the resolution.'))
    parser.add_argument('--random_flip', action='store_true', help='whether to randomly flip images horizontally')

    """ Args about Model """
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--use_ema', action='store_true', default='whether to use ema model for inference')

    return parser.parse_args()

args = parse_args()
config = OmegaConf.load(args.config)
pl.seed_everything(args.seed, workers=True)
config.name = args.name
config.savedir = args.savedir
config.mode = args.mode
config.datasets = args.dataset
config.batch_size = args.batch_size
config.center_crop = args.center_crop
config.random_flip = args.random_flip

trainer_model = StableDiffusionTrainer(config.ddconfig)
if args.resume == '':
    warnings.warn('No resume file specified')
elif os.path.exists(args.resume):
    if os.path.isdir(args.resume):
        filename = os.path.join(args.resume, 'model.pth')
        dirname = args.resume
    else:
        filename = args.resume
        dirname = os.path.dirname(args.resume)
    state_dict = torch.load(filename, map_location='cpu')
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        new_state_dict[sd_name.replace('_forward_module.', '')] = sd_param
    trainer_model.load_state_dict(new_state_dict, strict=False)
    print(filename, 'loaded')
else:
    raise FileNotFoundError(args.resume, 'not found')
if args.use_ema:
    trainer_model.ema_model.copy_to(trainer_model.unet)
trainer_model.freeze()
trainer_model.scheduler = DPMSolverMultistepScheduler.from_config(trainer_model.scheduler.config)

propmts = [
    "tourist boat for excursions on the tiber river, with bridges and panoramic portholes to appreciate the city from the river",
    "Aerial, close up: flying above the semi truck carrying freight cattle container for live animal land transportation. truck in united states hauling livestock box along the country highway at dusk",
    "street crossing with a traffic, walking people and a rainbow crosswalk in castro district. aerial video recording. editorial.",
    "Medicago sativa, alfalfa, lucerne in bloom - close up.  alfalfa is the most cultivated forage legume in the world and has been used as an herbal medicine since ancient times.",
    "Background hexagon texture, wax honeycomb from bee hive filled with golden honey. honeycomb consisting of macro overview beeswax, yellow sweet honey from bee beehive. honey nectar of bees honeycombs.",
    "A knight riding on a horse through the countryside",
    "A panda playing on a swing set", 
    "Sailboat sailing on a sunny day in a mountain lake, highly detailed",
    "A confused grizzly bear in calculus class",
    "A ballerina performs a beautiful and difficult dance on the roof of a very tall skyscraper; the city is lit up and glowing behind her"
]

propmts = [
    # Make-a-video/Damo/Ours
    'There is a table by a window with sunlight streaming through illuminating a pile of books.',
    'A blue unicorn flying over a mystical land.',
    'A dog wearing a Superhero outfit with red cape flying through the sky.',
    'Clown fish swimming through the coral reef.',
    'Robot dancing in times square.',
    'A litter of puppies running through the yard.',
    'A knight riding on a horse through the countryside.',
    'A panda playing on a swing set.',
    'A fluffy baby sloth with a knitted hat trying to figure out a laptop, close up, highly detailed, studio lighting, screen reflecting in its eyes.',
    'A teddy bear painting a portrait.',
    'A young couple walking in a heavy rain.',
    'A bear driving a car.',
    'A beautiful scenery which leads to a jumpscare.',
    'Sailboat sailing on a sunny day in a mountain lake.',
    'A musk ox grazing on beautiful wildflowers.',
    'Two kangaroos are busy cooking dinner in a kitchen.',
    'A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.',
    'A spaceship being pulled into a blackhole.',
    'An artists brush painting on a canvas close up.',
    'An emoji of a baby panda wearing a red hat, blue gloves, green shirt, and blue pants.',
    'Cat watching tv with a remote in hand.',
    'Horse drinking water.',
    'Humans building a highway on mars.',
    'Hyper-realistic photo of an abandoned industrial site during a storm.',
    'A ufo hovering over aliens in a field.',
    'A golden retriever eating ice cream on a beautiful tropical beach at sunset.',
    'A ballerina performs a beautiful and difficult dance on the roof of a very tall skyscraper, the city is lit up and glowing behind her.',
    'Unicorns running along a beach.',

    # Imagen video/Damo/Ours
    'A giraffe underneath a microwave.',
    'A colorful professional animated logo for ’Imagen Video’ written using paint brush in cursive. Smooth video.',
    'Wooden figurine surfing on a surfboard in space.',
    'A panda bear driving a car.',
    'Balloon full of water exploding in extreme slow motion.',
    'Melting pistachio ice cream dripping down the cone.',
    'A british shorthair jumping over a couch.',
    'Coffee pouring into a cup.',
    'A small hand-crafted wooden boat taking off to space.',
    'Bird-eye view of a highway in Los Angeles.',
    'Drone flythrough interior of Sagrada Familia cathedral.',
    'Wooden figurine walking on a treadmill made out of exercise mat.',
    'An astronaut riding a horse.',
    'A happy elephant wearing a birthday hat walking under the sea.',
    'Studio shot of minimal kinetic sculpture made from thin wire shaped like a bird on white background.',
    'Campfire at night in a snowy forest with starry sky in the background.',
    'A goldendoodle playing in a park by a lake.',
    'Incredibly detailed science fiction scene set on an alien planet, view of a marketplace. Pixel art.',
    'Pouring latte art into a silver cup with a golden spoon next to it.',
    'Shoveling snow.',
    'Drone flythrough of a tropical jungle covered in snow.',
    'A beautiful sunrise on mars, Curiosity rover. High definition, timelapse, dramatic colors.',
    'A shark swimming in clear Carribean ocean.',
    'A hand lifts a cup.',
    'A cat eating food out of a bowl, in style of Van Gogh.',
    'A tiddy bear washing dishes.',
    'Tiny plant sprout coming out of the ground.',
    'A sheep to right of wine glass.',
    'A shooting star flying through the night sky over mountains.',
    'A clear wine glass with turquoise-colored waves inside it.',
    'A swarm of bees flying around their hive.',
    'View of a castle with fantastically high towers reaching into the clouds in a hilly forest at dawn.',
    'A video of the Earth rotating in space.',
    'A bicycle on top of boat.',
    'An umbrella on top of a spoon.',
    'Sprouts in the shape of text ‘Imagen Video’ coming out of a fairytale book.',

    # Text from imagen video
    'A person riding a bike in the sunset.',
    'Origami dancers in white paper, 3D render, ultra-detailed, on white background, studio shot, dancing modern dance.',
    'A group of people hiking in a forest.',

    # Zero/Damo/ours
    'a horse galloping on a street',
    'a panda is playing guitar on times square',
    'a high quality realistic photo of a cute cat running in a beautiful meadow',
    'an astronaut is skiing down a hill',
    'a dog is walking down the street',
    'a panda dancing in Antarctica',
    'a man is running in the snow',
    'a man is riding a bicycle in the sunshine',
]

propmts = [
    # Make-a-video/Damo/Ours
    'There is a table by a window with sunlight streaming through illuminating a pile of books.',
    'A blue unicorn flying over a mystical land.',
    'A dog wearing a Superhero outfit with red cape flying through the sky.',
    'Clown fish swimming through the coral reef.',
    'Robot dancing in times square.',
    'A litter of puppies running through the yard.',
    'A knight riding on a horse through the countryside.',
    'A panda playing on a swing set.',
    'A fluffy baby sloth with a knitted hat trying to figure out a laptop, close up, highly detailed, studio lighting, screen reflecting in its eyes.',
]
propmts = [
    'A panda is driving a car, 4K, high definition',
    'Coffee pouring into a cup, 4K, high definition'
]

propmts = [
    'a young beautiful anime girl with long hair, wearing a white dress, anime style, highly detailed',
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer_model.half()
trainer_model.to(device)

save_dir = os.path.join(dirname, 'test_samples')
os.makedirs(save_dir, exist_ok=True)
for i, propmt in tqdm(enumerate(propmts), total=len(propmts)):
    result = trainer_model.inference(text=propmt, num_inference_steps=50, guidance_scale=9.0, use_dataset='webvid')

    video64 = ((result['video'] + 1) * 127.5).clamp(0, 255).to(torch.uint8)[0].permute(0, 2, 3, 1).cpu() # (t, h, w, c)
    video64 = video64.numpy()

    video_fps=8
    frame_gap=0

    imgs = [img for img in video64]
    video_clip = ImageSequenceClip(imgs, fps=video_fps/(frame_gap+1))
    video_clip.write_videofile(os.path.join(save_dir, f"{i}.mp4"), video_fps/(frame_gap+1), audio=False)
