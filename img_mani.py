from PIL import Image

class convolution(object):
    def __init__(self, filter, img_add):
        self.filter = filter
        self.img_add = img_add
        self.F = len(filter)  # assuming square filter
        self.stride = 1

    def open_img(self):
        self.org = Image.open(self.img_add)
        self.wid, self.height = self.org.size
        self.dims = self.wid  # assuming square image
    
    def apply(self):
        self.pixels = self.org.load()
        if (self.dims - self.F) % self.stride == 0:
            output_size = (self.dims - self.F) // self.stride + 1
            self.new = Image.new('RGB', (int(output_size), int(output_size)), 'white')
            for i in range(0, self.dims - self.F + 1, self.stride):
                for j in range(0, self.dims - self.F + 1, self.stride):
                    dot = [0, 0, 0]
                    for ir in range(self.F):
                        for ic in range(self.F):
                            pixel = self.pixels[i + ir, j + ic]
                            dot[0] += pixel[0] * self.filter[ir][ic]
                            dot[1] += pixel[1] * self.filter[ir][ic]
                            dot[2] += pixel[2] * self.filter[ir][ic]
                    dot = [int(c) for c in dot]  # Convert dot values to integers
                    self.new.putpixel((i // self.stride, j // self.stride), tuple(dot))
            self.new.save("cat_face_pixel_art_random.png")
            print("done")

# Define the blur filter
Reg_blur = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
Gaussian_blur = [
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
]
random_filter = [
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
]

# Create the convolution object with the image path
conv1 = convolution(random_filter, "cat_face_pixel_art.png")

# Open the image
conv1.open_img()

# Apply the filter
conv1.apply()

# Save the new image

