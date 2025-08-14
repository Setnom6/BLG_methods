import os
import glob
import imageio.v2 as imageio

current_dir = os.path.dirname(os.path.abspath(__file__))

base_name = "Heatmap"
extension = ".png"
output_video = os.path.join(current_dir, "figures", "heatmap.mp4")
fps = 2

pattern = os.path.join(current_dir, "figures", "forVideo", f"{base_name}*{extension}")
print(pattern)
png_files = sorted(glob.glob(pattern))

if not png_files:
    print("No png files founded.")
    exit()

print(f"generating video with {len(png_files)} frames...")

with imageio.get_writer(output_video, fps=fps) as writer:
    for file in png_files:
        image = imageio.imread(file)
        writer.append_data(image)

print(f"Video saved as {output_video}")

