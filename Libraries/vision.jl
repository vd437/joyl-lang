// vision.joyl - Professional image processing
pub struct Image {
    pixels: Tensor,
    height: int,
    width: int,
    channels: int
}

impl Image {
    /// Load image from file
    pub fn open(path: string) -> Image {
        let raw = native_image_load(path);
        
        // Convert to standardized format (RGB)
        let pixels = match raw.channels {
            1 => raw.pixels.repeat(3),  // Grayscale to RGB
            3 => raw.pixels,
            4 => raw.pixels.remove_alpha(),  // RGBA to RGB
            _ => panic("Unsupported image format")
        };
        
        Image {
            pixels: Tensor::from_data(pixels, [raw.height, raw.width, 3]),
            height: raw.height,
            width: raw.width,
            channels: 3
        }
    }

    /// Resize image with bilinear interpolation
    pub fn resize(&self, new_height: int, new_width: int) -> Image {
        let scale_y = self.height as float / new_height as float;
        let scale_x = self.width as float / new_width as float;
        
        let mut output = Tensor::zeros([new_height, new_width, 3]);
        
        // Parallel bilinear sampling
        parallel_for y in 0..new_height {
            for x in 0..new_width {
                let src_y = (y as float * scale_y).clamp(0, self.height-1);
                let src_x = (x as float * scale_x).clamp(0, self.width-1);
                
                let y1 = src_y.floor() as int;
                let x1 = src_x.floor() as int;
                let y2 = (y1 + 1).min(self.height-1);
                let x2 = (x1 + 1).min(self.width-1);
                
                let dy = src_y - y1 as float;
                let dx = src_x - x1 as float;
                
                for c in 0..3 {
                    let val = lerp(
                        lerp(self.pixels.get([y1, x1, c]), 
                             self.pixels.get([y1, x2, c]), dx),
                        lerp(self.pixels.get([y2, x1, c]), 
                             self.pixels.get([y2, x2, c]), dx),
                        dy
                    );
                    output.set([y, x, c], val);
                }
            }
        }
        
        Image {
            pixels: output,
            height: new_height,
            width: new_width,
            channels: 3
        }
    }
}