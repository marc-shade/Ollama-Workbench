from PIL import Image, ImageDraw
import os

def generate_llama_icon(size):
    # Create a new image with a white background
    image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Calculate dimensions
    padding = size // 4
    width = size - (2 * padding)
    height = size - (2 * padding)
    
    # Draw a simple llama silhouette
    # Body
    body_points = [
        (padding + width//4, padding + height//2),  # Top of body
        (padding + width*3//4, padding + height//2),  # Back
        (padding + width*3//4, padding + height*3//4),  # Back leg
        (padding + width//4, padding + height*3//4),  # Front leg
    ]
    
    # Neck and head
    neck_points = [
        (padding + width//4, padding + height//2),  # Bottom of neck
        (padding + width//4, padding + height//4),  # Top of neck
        (padding + width//3, padding + height//5),  # Head
    ]
    
    # Draw the llama in a nice blue color
    draw.polygon(body_points, fill=(66, 133, 244, 255))  # Google Blue
    draw.line(neck_points, fill=(66, 133, 244, 255), width=size//8)
    
    # Add ear
    ear_size = size//8
    draw.ellipse([
        padding + width//3 - ear_size//2,
        padding + height//5 - ear_size//2,
        padding + width//3 + ear_size//2,
        padding + height//5 + ear_size//2
    ], fill=(66, 133, 244, 255))
    
    return image

def main():
    # Create icons directory if it doesn't exist
    os.makedirs('icons', exist_ok=True)
    
    # Generate icons of different sizes
    sizes = [16, 48, 128]
    for size in sizes:
        icon = generate_llama_icon(size)
        icon.save(f'icons/icon{size}.png')
        print(f'Generated icon{size}.png')

if __name__ == '__main__':
    main()
