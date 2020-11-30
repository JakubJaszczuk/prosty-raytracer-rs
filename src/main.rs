use std::fs::File;
use std::io::{BufReader, BufRead, Write};
use rand::prelude::*;

type Vec2 = nalgebra::Vector2<f32>;
type Vec3 = nalgebra::Vector3<f32>;
type Uvec3 = nalgebra::Vector3<u32>;

//////////////////////////////////////////////////

struct HitData {
    uv: nalgebra::Vector2<f32>,
    object_index: usize,
    triangle_index: usize,
    t: f32,
}

impl Default for HitData {
    fn default() -> Self {
        HitData {
            uv: nalgebra::zero(),
            object_index: usize::default(),
            triangle_index: usize::default(),
            t: f32::default()
        }
    }
}

impl HitData {
    pub fn new(uv: Vec2, dist: f32, object: usize, triangle: usize) -> Self {
        HitData {
            uv, object_index: object, triangle_index: triangle, t: dist
        }
    }
}

//////////////////////////////////////////////////

pub struct Camera {
    position: Vec3,
    fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {position: nalgebra::zero(), fov: 60.0}
    }
}

impl Camera {
    pub fn new(position: Vec3, fov: f32) -> Self {
        Camera {position, fov}
    }

    pub fn from_fov(fov: f32) -> Self {
        Camera {fov, ..Default::default()}
    }

    pub fn from_pos(position: Vec3) -> Self {
        Camera {position, ..Default::default()}
    }
}

//////////////////////////////////////////////////

enum Material {
    Emission(Vec3, f32),
    Diffuse(Vec3),
    Glossy(Vec3, f32)
}

//////////////////////////////////////////////////

pub struct Object {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    triangles: Vec<Uvec3>,
    material: Material,
    aabb1: Vec3,
    aabb2: Vec3,
}

impl Default for Object {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            triangles: Vec::new(),
            material: Material::Diffuse(Vec3::new(0.8, 0.8, 0.8)),
            aabb1: Vec3::new(0.0, 0.0, 0.0),
            aabb2: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}

impl Object {
    pub fn create_aabb(&mut self) {
        self.aabb1 = self.vertices[0];
        self.aabb1 = self.vertices[1];
        for v in self.vertices.iter() {
            if v.x < self.aabb1.x {self.aabb1.x = v.x}
            if v.y < self.aabb1.y {self.aabb1.y = v.y}
            if v.z < self.aabb1.z {self.aabb1.z = v.z}
            if v.x > self.aabb2.x {self.aabb2.x = v.x}
            if v.y > self.aabb2.y {self.aabb2.y = v.y}
            if v.z > self.aabb2.z {self.aabb2.z = v.z}
        }
    }

    pub fn load_from_file(&mut self, path: &str) {
        let mut ilosc_v: u32 = 0;
        let mut ilosc_t: u32 = 0;
        let file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return
        };
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            if line.starts_with("element vertex ") {
                ilosc_v = line[15..].trim().parse().unwrap();
            }
            if line.starts_with("element face ") {
                ilosc_t = line[13..].trim().parse().unwrap();
            }
            else if line.starts_with("end_header") 
            {
                line.clear();
                break;
            }
            line.clear();
        }
        let ilosc_v = ilosc_v;
        let ilosc_t = ilosc_t;
        for _ in 0..ilosc_v {
            reader.read_line(&mut line).unwrap();
            let v: Vec<f32> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
            self.vertices.push(Vec3::new(v[0], v[1], v[2]));
            self.normals.push(Vec3::new(v[3], v[4], v[5]));
            line.clear();
        }
        for _ in 0..ilosc_t {
            reader.read_line(&mut line).unwrap();
            let v: Vec<u32> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
            self.triangles.push(Uvec3::new(v[1], v[2], v[3]));
            line.clear();
        }
        self.create_aabb();
    }
}

//////////////////////////////////////////////////
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Default for Ray {
    fn default() -> Self {
        Ray {origin: Vec3::zeros(), dir: Vec3::new(1.0, 0.0, 0.0)}
    }
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        Ray {origin, dir}
    }

    pub fn random_direction() -> Vec3 {
        let dis = rand::distributions::Uniform::new(-1.0, 1.0);
        let mut r = rand::thread_rng();
        Vec3::new(dis.sample(&mut r), dis.sample(&mut r), dis.sample(&mut r)).normalize()
    }

    pub fn slabs(&self, p0: &Vec3, p1: &Vec3) -> bool {
        let inv_ray_dir = self.dir.map(|x| 1.0 / x);
        let t0 = (p0 - self.origin).component_mul(&inv_ray_dir);
        let t1 = (p1 - self.origin).component_mul(&inv_ray_dir);
        let (tmin, tmax) = nalgebra::Matrix::inf_sup(&t0, &t1);
        tmin.max() <= tmax.min()
    }

    fn intersect_ray_triangle(&self, vert0: &Vec3, vert1: &Vec3, vert2: &Vec3, bary_position: &mut Vec2, distance: &mut f32) -> bool {
        let edge1 = vert1 - vert0;
        let edge2 = vert2 - vert0;
        let p = self.dir.cross(&edge2);
        let det = edge1.dot(&p);
        let perpendicular;

        if det > std::f32::EPSILON {
            let dist = self.origin - vert0;
            bary_position.x = dist.dot(&p);
            if bary_position.x < 0.0 || bary_position.x > det {
                return false;
            }
            perpendicular = dist.cross(&edge1);
            bary_position.y = self.dir.dot(&perpendicular);
            if (bary_position.y < 0.0) || ((bary_position.x + bary_position.y) > det) {
                return false;
            }
        }
        else if det < -std::f32::EPSILON {
            let dist = self.origin - vert0;
            bary_position.x = dist.dot(&p);
            if bary_position.x > 0.0 || bary_position.x < det {
                return false;
            }
            perpendicular = dist.cross(&edge1);
            bary_position.y = self.dir.dot(&perpendicular);
            if (bary_position.y > 0.0) || ((bary_position.x + bary_position.y) < det) {
                return false;
            }
        }
        else {
            return false;
        }

        let inv_det = 1.0 / det;
        *distance = edge2.dot(&perpendicular) * inv_det;
        *bary_position *= inv_det;
        
        true
    }

    pub fn closest_intersection(&self, scene_data: &[Object]) -> HitData {
        let mut closest_uv = Vec2::new(0.0, 0.0);
        let mut closest_t = std::f32::MAX;
        let mut tri = std::usize::MAX;
        let mut obj = std::usize::MAX;

        let mut uv: Vec2 = Vec2::new(0.0, 0.0);
        let mut t = std::f32::MAX;
        for (i, object) in scene_data.iter().enumerate() {
            if self.slabs(&object.aabb1, &object.aabb2) {
                for (j, triangle) in object.triangles.iter().enumerate() {
                    let b = self.intersect_ray_triangle(&object.vertices[triangle[0] as usize], &object.vertices[triangle[1] as usize], &object.vertices[triangle[2] as usize], &mut uv, &mut t);
                    if b && t > 0.0 && t < closest_t {
                        closest_t = t;
                        closest_uv = uv;
                        tri = j;
                        obj = i;
                    }
                }
            }
        }

        HitData::new(closest_uv, closest_t, obj, tri)
    }
}

//////////////////////////////////////////////////
struct LightRay {
    ray: Ray,
    ttl: u8,
}

impl Default for LightRay {
    fn default() -> Self {
        LightRay {ray: Ray::default(), ttl: 4}
    }
}

impl LightRay {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        LightRay {ray: Ray::new(origin, direction), ttl: 4}
    }
    
    pub fn new_with_ttl(origin: Vec3, direction: Vec3, max_depth: u8) -> Self {
        LightRay {ray: Ray::new(origin, direction), ttl: max_depth}
    }

    pub fn trace(&self, scene_data: &[Object], rays_count: u32, directions: &[Vec3]) -> Vec3 {
        let hit_data = self.ray.closest_intersection(scene_data);
        if hit_data.object_index == std::usize::MAX {
            return Vec3::zeros();
        }

        let obj = &scene_data[hit_data.object_index];
        let mat = &obj.material;

        match mat {
            Material::Emission(color, power) => color * *power,
            Material::Glossy(_, _) => Vec3::new(0.0, 0.0, 0.0),
            Material::Diffuse(diffuse_color) => {
                if self.ttl == 0 {
                    return Vec3::new(0.0, 0.0, 0.0);
                }
                let tri: &Uvec3 = &obj.triangles[hit_data.triangle_index];
                let normal = barycentric_conversion(hit_data.uv, &obj.normals[tri[0] as usize], &obj.normals[tri[1] as usize], &obj.normals[tri[2] as usize]).normalize();
                let orig = barycentric_conversion(hit_data.uv, &obj.vertices[tri[0] as usize], &obj.vertices[tri[1] as usize], &obj.vertices[tri[2] as usize]) + normal * 0.00001;
                //let orig = self.ray.origin + hit_data.t * self.ray.dir + normal * 0.00001f32;
                let mut counter: u32 = 0;
                let mut i = 0;
                let mut color: Vec3 = Vec3::zeros();
                // TODO
                while counter < rays_count {
                    
                    let direction = if cfg!(feature = "RANDOM") {
                        Ray::random_direction()
                    }
                    else {
                        unsafe {
                            *directions.get_unchecked(i)
                        }
                    };
                    
                    //let direction = &directions[i];
                    /*
                    let direction = unsafe {
                        directions.get_unchecked(i)
                    };
                    */
                    let dot = normal.dot(&direction);
                    if dot > 0.0 {
                        color += LightRay::new_with_ttl(orig, direction, self.ttl - 1).trace(scene_data, rays_count, directions) * dot;
                        counter += 1;
                    }
                    i += 1;
                }
                //color.apply(|e| e / counter as f32);
                //diffuse_color.component_mul(&color)
                diffuse_color.component_mul(&color.map(|e| e / counter as f32))
            } 
        }
    }
}

#[inline]
fn barycentric_conversion(uv: Vec2, p0: &Vec3, p1: &Vec3, p2: &Vec3) -> Vec3 {
    p0 * (1.0 - (uv.x + uv.y)) + p1 * uv.x + p2 * uv.y
}

//////////////////////////////////////////////////

mod render {
    use super::*;
    type U8Vec3 = nalgebra::Vector3<u8>;

    fn pixel_coord(fov: f32, resolution: u32, i: u32, j: u32) -> Vec2 {
        let half_screen_length = (fov / 2.0).to_radians().tan();
        let half_pixel_length = half_screen_length / resolution as f32;
        //-half_screen_length + half_pixel_length + 2.0 * half_pixel_length * Vec2::new(i as f32, j as f32)
        (2.0 * half_pixel_length * Vec2::new(i as f32, j as f32)).map(|x| x + (-half_screen_length + half_pixel_length))
    }

    fn create_primary_ray(camera: &Camera, resolution: u32, i: u32, j: u32, ray_depth: u8) -> LightRay {
        let coord = pixel_coord(camera.fov, resolution, i, j);
        let dir = Vec3::new(coord.x, 1.0, coord.y).normalize();
        LightRay::new_with_ttl(camera.position, dir, ray_depth)
    }

    fn render(scene_data: &[Object], camera: &Camera, resolution: u32, buffer: &mut[U8Vec3], stride: u32, rays_count: u32, ray_depth: u8, directions: &[Vec3]) {
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let ind: u32 = stride + i as u32;
            let light_ray = create_primary_ray(camera, resolution, ind % resolution, resolution - (ind / resolution), ray_depth);
            let color = light_ray.trace(scene_data, rays_count, directions);
            //if i % 4000 == 0 {
                //println!("{}", ind);
            //}
            *pixel = color_conversion(&color);
        }
    }

    pub fn run(scene_data: &[Object], camera: &Camera, resolution: u32, rays_count: u32, ray_depth: u8) {
        let directions: Vec<Vec3> = (0..(rays_count * 100)).map(|_| Ray::random_direction()).collect();

        let mut buffer: Vec<U8Vec3> = vec![U8Vec3::zeros(); (resolution * resolution) as usize];
        let concurent = num_cpus::get();
        
        crossbeam::thread::scope(|s| {
            let mut threads = Vec::with_capacity(concurent);
            let len = buffer.len() / concurent;
            let chunks = buffer.chunks_mut(len);
            let chunks_len = chunks.len();
            let f = &directions;

            for (i, chunk) in chunks.enumerate() {
                threads.push(s.builder().name(format!("Watek - {}", i)).spawn(move |_| {
                    render(scene_data, camera, resolution, chunk, (i * len) as u32, rays_count, ray_depth, &f);
                }).unwrap());
            }
            println!("Chunks: {}", chunks_len);
            for thread in threads.into_iter() {
                let _ = thread.join();
            }
        }).unwrap();

        println!("Zapisuję do pliku.");
        let mut file = File::create("imgs/img").unwrap();
        write!(file, "P3\n{} {}\n255\n", resolution, resolution).unwrap();
        for color in buffer.iter() {
            write!(file, "{} {} {} ", color.x, color.y, color.z).unwrap();
        }
    }

    fn color_conversion(color: &Vec3) -> U8Vec3 {
        color.map(|e| (nalgebra::clamp(e, 0.0, 1.0) * 254.0) as u8)
    }
}

//////////////////////////////////////////////////

fn main() {
    println!("Hello, world!");
    if cfg!(feature = "RANDOM") {
        println!("RANDOM mode activated!");
    }
    else {
        println!("PREGENERATE mode activated!");
    };

	// Można ustawić rozdzielczość z linii poleceń
    let argv: Vec<_> = std::env::args().collect();
    let argc = argv.len();
	let res: u32 = if argc > 1 {
		argv[1].parse().unwrap()
    }
    else {
        512
    };
	let rays_count: u32 = if argc > 2 {
		argv[2].parse().unwrap()
    }
    else {
        20
    };
	let ray_depth: u8 = if argc > 3 {
		argv[3].parse().unwrap()
    }
    else {
        2
    };
    let res = res;
    println!("{} | {} | {}", res, rays_count, ray_depth);

    // Ustawienie sceny
	let mut cube1: Object = Default::default();
	cube1.load_from_file("scene/cube1.ply");
	let mut cube2: Object = Default::default();
	cube2.load_from_file("scene/cube2.ply");
	let mut back: Object = Default::default();
	back.load_from_file("scene/back.ply");
	let mut top: Object = Default::default();
	top.load_from_file("scene/top.ply");
	let mut bottom: Object = Default::default();
	bottom.load_from_file("scene/bottom.ply");
	let mut left: Object = Default::default();
	left.load_from_file("scene/left.ply");
	let mut right: Object = Default::default();
	right.load_from_file("scene/right.ply");
	let mut light: Object = Default::default();
    light.load_from_file("scene/light.ply");

    cube1.material = Material::Diffuse(Vec3::new(0.8, 0.5, 0.5));
	
	cube2.material = Material::Diffuse(Vec3::new(0.3, 1.0, 0.8));

	left.material = Material::Diffuse(Vec3::new(1.0, 0.0, 0.0));

	right.material = Material::Diffuse(Vec3::new(0.0, 1.0, 0.0));

	back.material = Material::Diffuse(Vec3::new(0.9, 0.9, 0.9));

	top.material = Material::Diffuse(Vec3::new(0.9, 0.9, 0.9));

	bottom.material = Material::Diffuse(Vec3::new(0.9, 0.9, 0.9));

    light.material = Material::Emission(Vec3::new(1.0, 1.0, 1.0), 30.0);
    
    let mut scene: Vec<Object> = Vec::new();
	scene.push(light);
	scene.push(cube1);
	scene.push(cube2);
	scene.push(back);
	scene.push(left);
	scene.push(right);
	scene.push(top);
    scene.push(bottom);

    let camera = Camera::new(Vec3::new(0.0, -3.0, 0.0), 55.0);
    render::run(&scene, &camera, res, rays_count, ray_depth);

    // RUSTFLAGS="-C target-cpu=native" cargo run --release --features "RANDOM" -- 256 2 2
}
