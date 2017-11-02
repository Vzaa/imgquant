extern crate image;
extern crate rand;
extern crate rayon;

use std::env;
use std::fs::File;

use rayon::prelude::*;
use rayon::iter::ParallelIterator;
use rand::{thread_rng, Rng};
use image::{ImageBuffer, Rgb};

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


pub struct KMeans<T>
where
    T: Sized,
    T: Copy,
    T: rand::Rand,
{
    vals: Vec<[T; 3]>,
}


impl<T> KMeans<T>
where
    T: Copy,
    f32: From<T>,
    T: rand::Rand,
{
    pub fn new(k: usize) -> KMeans<T> {
        let mut rng = thread_rng();
        let mut vals = Vec::new();

        for _ in 0..k {
            vals.push([rng.gen(), rng.gen(), rng.gen()]);
        }

        KMeans { vals }
    }

    pub fn class_val(&self, p: &[T]) -> &[T; 3] {
        let idx = self.class_idx(p);
        &self.vals[idx]
    }

    pub fn class_idx(&self, p: &[T]) -> usize {
        let m = self.vals
            .iter()
            .map(|k| dist(k, p))
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        m.unwrap().0
    }
}

impl KMeans<u8> {
    pub fn update(&mut self, imgrgb: &ImgRgb) {
        let new_centers: Vec<_> = (0..self.vals.len())
            .into_par_iter()
            .map(|c| {
                let (cnt, sums) = imgrgb
                    .pixels()
                    .filter(|&p| self.class_idx(&p.data) == c)
                    .map(|p| [p.data[0] as u64, p.data[1] as u64, p.data[2] as u64])
                    .fold((0, [0, 0, 0]), |(cnt, acc), x| {
                        (cnt + 1, [acc[0] + x[0], acc[1] + x[1], acc[2] + x[2]])
                    });

                if cnt == 0 {
                    return None;
                }

                Some([
                    (sums[0] / cnt) as u8,
                    (sums[1] / cnt) as u8,
                    (sums[2] / cnt) as u8,
                ])
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
    let k = env::args()
        .nth(2)
        .unwrap_or("12".to_owned())
        .parse()
        .unwrap();

    let iters = env::args()
        .nth(3)
        .unwrap_or("20".to_owned())
        .parse()
        .unwrap();

    let mut img = image::open(&filename).unwrap();

    {
        let mut kmeans = KMeans::new(k);
        let imgrgb = img.as_mut_rgb8().expect("Cannot read image as RGB");

        // Iterate a fixed amount
        for _ in 0..iters {
            kmeans.update(imgrgb);
        }

        // Quantize the image
        for p in imgrgb.pixels_mut() {
            let v = kmeans.class_val(&p.data);
            *p = Pix { data: *v };
        }
    }

    let mut fout = File::create("qout.png").unwrap();
    img.save(&mut fout, image::PNG).unwrap();
}
