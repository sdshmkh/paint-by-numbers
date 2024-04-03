# Paint by Numbers

Paint by Numbers is a versatile command-line tool that allows users to either create a new image using DALL·E or edit an existing image based on a given prompt. The tool is designed to be intuitive and easy to use, offering a variety of options to customize the image creation and editing process.

## Features

- **Mode Selection**: Choose between creating a new image (`C`) or editing an existing one (`E`).
- **Image Path**: Specify the path to the image you wish to edit.
- **Prompt**: Provide a descriptive prompt to guide the creation or editing of the image.
- **Palette Count**: Define the number of colors for your palette, enhancing the visual appeal and clarity of the image.

## Setup

Add a `.env` file and add your `OPENAI_API_KEY` in the `.env` file. 

Install `virtualenv` via pip. It's recommended to update pip to the latest version before doing so.

```
pip install virtualenv
```
Create a new virtualenv with 
```
python -m virtualenv venv
```
Activate the venv
```
venv\Scripts\activate 
```
Run 
```
pip install -r requirements.txt
``` 

## Usage

To use Paint by Numbers, you can follow the command-line syntax described below. This tool supports various options to tailor the image processing according to your needs.

```
Usage: paint_by_numbers.py [OPTIONS]

Options:
  --mode TEXT              create a new image using Dall-E or a variation of
                           an existing image, options - v/C
  --image_path TEXT        path to the image
  --prompt TEXT            prompt to create/edit the image
  --palette_count INTEGER  number of colors for your palette
  --help                   Show this message and exit.
```

```
python3 paint_by_numbers.py --image_path='shapes.jpeg'

```

```
python3 paint_by_numbers.py --mode='v' --image_path='shapes.jpeg'
```

```
python3 paint_by_numbers.py --mode='C' --prompt='image of NYC skyline'
```
