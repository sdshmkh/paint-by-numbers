import click
from dotenv import load_dotenv
import cv2

from model import create
from model import trigger_dalle

@click.command()
@click.option('--mode', default=None, help='create a new image using Dall-E or a variation of an existing image, options - v/C')
@click.option('--image_path', default=None, help='path to the image')
@click.option('--prompt', default=None, help='prompt to create/edit the image')
@click.option('--palette_count', default=10, help='number of colors for your palette')
def run(mode, image_path, prompt, palette_count):
    load_dotenv()
    click.echo("Welcome to Paint By Numbers")
    if mode == 'C' and not prompt:
        click.echo("Please enter a prompt")
        return 
    if mode == 'v' and not image_path:
        click.echo("Please enter a path")
        return
    
    image_path = trigger_dalle(image_path, prompt, mode)
    pbk_image_path, pbk_image = create(image_path, n_clusters=palette_count)
    click.echo(pbk_image_path)

    cv2.imshow('Image Title', pbk_image)

    # Wait for any key to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    run()