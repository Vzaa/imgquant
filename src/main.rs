extern crate image;
extern crate rand;
extern crate rayon;

use std::env;
use std::fs::File;

use rayon::prelude::*;
use rand::{thread_rng, Rng};
use image::{ImageBuffer, Rgb};

use rayon::iter::ParallelIterator;

pub type Pix = Rgb<u8>;
pub type ImgRgb = ImageBuffer<Pix, Vec<u8>>;

fn dist<T>(a: &[T], b: &[T]) -> f32
where
    f32: From<T>,
    T: Copy,
{
    a.iter().zip(b.iter()).fold(0.0_f32, |acc, (x, y)| {
        let fx: f32 = (*x).into();
        let fy: f32 = (*y).into();

        let d = (fx - fy).abs();
        (d.powi(2) + acc.powi(2)).sqrt()
    })
}


struct KMeans<T> {
    vals: Vec<T>,
}

impl KMeans<Pix> {
    pub fn new(cnt: usize) -> KMeans<Pix> {
        let mut rng = thread_rng();

        let mut vals = Vec::new();

        for _ in 0..cnt {
            vals.push(Pix {
                data: [rng.gen(), rng.gen(), rng.gen()],
            });
        }

        KMeans { vals }
    }

    pub fn class(&self, p: &Pix) -> usize {
        let m = self.vals
            .iter()
            .map(|k| dist(&k.data, &p.data))
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        m.unwrap().0
    }

    pub fn update(&mut self, imgrgb: &ImgRgb) {
        let new_centers: Vec<_> = (0..self.vals.len())
            .into_par_iter()
            .map(|c| {
                let (cnt, sums) = imgrgb
                    .pixels()
                    .filter(|&p| self.class(p) == c)
                    .map(|p| [p.data[0] as u64, p.data[1] as u64, p.data[2] as u64])
                    .fold((0, [0, 0, 0]), |(cnt, acc), x| {
                        (cnt + 1, [acc[0] + x[0], acc[1] + x[1], acc[2] + x[2]])
                    });

                if cnt == 0 {
                    return None;
                }

                Some(Pix {
                    data: [
                        (sums[0] / cnt) as u8,
                        (sums[1] / cnt) as u8,
                        (sums[2] / cnt) as u8,
                    ],
                })
            })
            .collect();

        for (o, &n) in self.vals.iter_mut().zip(new_centers.iter()) {
            if let Some(v) = n {
                *o = v;
            }
        }
    }
}

fn main() {
    let filename = env::args().nth(1).expect("No filename entered");
    let mut img = image::open(&filename).unwrap();

    {
        let mut kmeans = KMeans::new(12);
        let imgrgb = img.as_mut_rgb8().expect("Cannot read image as RGB");

        // Iterate a fixed amount
        for _ in 0..20 {
            kmeans.update(imgrgb);
        }

        // Quantize the image
        for p in imgrgb.pixels_mut() {
            let c = kmeans.class(&p);
            *p = kmeans.vals[c];
        }
    }

    let mut fout = File::create("qout.png").unwrap();
    img.save(&mut fout, image::PNG).unwrap();
}
