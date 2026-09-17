from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

# Function to draw an arrow on the image
def draw_arrow(draw, x, y, theta, length=20):
    # Convert theta to radians
    theta_rad = math.radians(theta)
    
    # Calculate arrow tip position
    arrow_x = x + length * math.cos(theta_rad)
    arrow_y = y - length * math.sin(theta_rad)  # Subtract to make the angle consistent with image coordinate system
    
    # Draw the arrow (line from point (x, y) to (arrow_x, arrow_y))
    draw.line([x, y, arrow_x, arrow_y], fill="red", width=2)
    
    # Optionally, draw an arrowhead
    arrow_head_angle = math.radians(15)
    arrow_head_length = 5
    arrow_draw_angle_1 = math.atan2(arrow_y - y, arrow_x - x) + arrow_head_angle
    arrow_draw_angle_2 = math.atan2(arrow_y - y, arrow_x - x) - arrow_head_angle
    
    arrow_x1 = arrow_x + arrow_head_length * math.cos(arrow_draw_angle_1)
    arrow_y1 = arrow_y - arrow_head_length * math.sin(arrow_draw_angle_1)
    arrow_x2 = arrow_x + arrow_head_length * math.cos(arrow_draw_angle_2)
    arrow_y2 = arrow_y - arrow_head_length * math.sin(arrow_draw_angle_2)
    
    draw.line([arrow_x, arrow_y, arrow_x1, arrow_y1], fill="red", width=2)
    draw.line([arrow_x, arrow_y, arrow_x2, arrow_y2], fill="red", width=2)

# Function to process the text file and mark the points on the image
def mark_points_on_image(image_path, text_file):
    # Open the image
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format to prevent color changes
    draw = ImageDraw.Draw(image)
    
    # Read points and angles from the text file
    with open(text_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y, theta = map(int, line.split(','))
            
            # Draw a point (circle) on the image
            draw.ellipse((x-3, y-3, x+3, y+3), fill="blue", outline="blue")
            
            # Draw the arrow for theta
            draw_arrow(draw, x, y, theta)
    
    # Show the image with points and arrows, maintaining original color
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Example usage
image_path = "87_1.jpg"  # Provide the path to your image file
text_file = "87_1.txt"  # Provide the path to your text file containing the points and angles
mark_points_on_image(image_path, text_file)
