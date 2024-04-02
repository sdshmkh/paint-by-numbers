# Paint by Numbers

Paint by Numbers is a versatile command-line tool that allows users to either create a new image using DALL·E or edit an existing image based on a given prompt. The tool is designed to be intuitive and easy to use, offering a variety of options to customize the image creation and editing process.

## Features

- **Mode Selection**: Choose between creating a new image (`C`) or editing an existing one (`E`).
- **Image Path**: Specify the path to the image you wish to edit.
- **Prompt**: Provide a descriptive prompt to guide the creation or editing of the image.
- **Palette Count**: Define the number of colors for your palette, enhancing the visual appeal and clarity of the image.

## Usage

To use Paint by Numbers, you can follow the command-line syntax described below. This tool supports various options to tailor the image processing according to your needs.

```
Usage: paint_by_numbers.py [OPTIONS]

Options:
--mode TEXT create a new image using Dall-E or edit an existing
image, options - C for create, E for edit
--image_path TEXT path to the image you want to use/edit
--prompt TEXT prompt to guide the creation/editing of the image
--palette_count INTEGER number of colors for your palette
--help Show this message and exit.
```
