use std::env;

use image::{ImageBuffer, Rgb};
use itertools::izip;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub type Pix = Rgb<u8>;
pub type ImgRgb = ImageBuffer<Pix, Vec<u8>>;

fn euclidi<'a, T, U>(a: T, b: T) -> f32
where
    T: Iterator<Item = &'a U>,
    U: Into<f32>,
    U: Copy,
    U: 'a,
{
    izip!(a, b).fold(0.0_f32, |acc, (x, y)| {
        let fx: f32 = (*x).into();
        let fy: f32 = (*y).into();

        let d = (fx - fy).abs();
        (d.powi(2) + acc.powi(2)).sqrt()
    })
}

pub trait Dist {
    fn dist(self, other: Self) -> f32;
}

impl<'a, T, U> Dist for T
where
    T: IntoIterator<Item = &'a U>,
    U: Into<f32>,
    U: Copy,
    U: 'a,
{
    fn dist(self, other: Self) -> f32 {
        euclidi(self.into_iter(), other.into_iter())
    }
}

pub struct KMeans<T> {
    vals: Vec<T>,
}

impl<'a, T> KMeans<T>
where
    &'a T: Dist,
    T: 'a,
    Standard: Distribution<T>,
{
    pub fn new(k: usize) -> KMeans<T> {
        let mut rng = thread_rng();
        let mut vals = Vec::new();

        for _ in 0..k {
            vals.push(rng.gen());
        }

        KMeans { vals }
    }

    pub fn class_val(&'a self, p: &'a T) -> &T {
        let idx = self.class_idx(p);
        &self.vals[idx]
    }

    pub fn class_idx(&'a self, p: &'a T) -> usize {
        let m = self
            .vals
            .iter()
            .map(|k| k.dist(p))
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        m.unwrap().0
    }
}

impl KMeans<[u8; 3]> {
    pub fn update<F, I>(&mut self, ff: F)
    where
        F: Fn() -> I,
        I: Iterator<Item = [u8; 3]>,
        F: Send,
        F: Sync,
    {
        let new_centers: Vec<_> = (0..self.vals.len())
            .into_par_iter()
            .map(|c| {
                let (cnt, sums) = ff().filter(|&p| self.class_idx(&p) == c).fold(
                    (0, [0_u64; 3]),
                    |(cnt, mut acc), x| {
                        (cnt + 1, {
                            izip!(&mut acc, &x).for_each(|(a, b)| *a += u64::from(*b));
                            acc
                        })
                    },
                );

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

        for (o, &n) in izip!(&mut self.vals, &new_centers) {
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
        .unwrap_or_else(|| "12".to_owned())
        .parse()
        .unwrap();

    let iters = env::args()
        .nth(3)
        .unwrap_or_else(|| "20".to_owned())
        .parse()
        .unwrap();

    let outfl = env::args().nth(4).unwrap_or_else(|| "qout.png".to_owned());

    let mut img = image::open(&filename).unwrap();

    {
        let mut kmeans = KMeans::new(k);
        let imgrgb = img.as_mut_rgb8().expect("Cannot read image as RGB");

        // Iterate a fixed amount
        for _ in 0..iters {
            kmeans.update(|| imgrgb.pixels().map(|p| p.data));
        }

        // Quantize the image
        for p in imgrgb.pixels_mut() {
            p.data = *kmeans.class_val(&p.data);
        }
    }

    img.save(&outfl).unwrap();
}
