//! Orbit camera with Maya/Mudbox navigation semantics.

use glam::{Mat4, Vec2, Vec3};
use sculpt_core::geom::{Aabb, Ray};

#[derive(Clone, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { target: Vec3::ZERO, distance: 4.0, yaw: 0.0, pitch: 0.15, fov_y: 35f32.to_radians() }
    }
}

impl Camera {
    pub fn frame(&mut self, b: &Aabb) {
        if b.is_empty() {
            return;
        }
        self.target = b.center();
        let r = b.diagonal() * 0.5;
        self.distance = r / (self.fov_y * 0.5).sin() * 1.1;
    }

    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos());
        self.target + dir * self.distance
    }

    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or(Vec3::X)
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.01).max(1e-4);
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, self.distance * 100.0)
    }

    /// Ray through a point given in viewport pixels (origin top-left).
    pub fn ray(&self, px: Vec2, size: Vec2) -> Ray {
        let ndc = Vec2::new(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0);
        let half_h = (self.fov_y * 0.5).tan();
        let half_w = half_h * size.x / size.y.max(1.0);
        let dir = self.forward() + self.right() * (ndc.x * half_w) + self.up() * (ndc.y * half_h);
        Ray::new(self.eye(), dir)
    }

    /// World units covered by one pixel at the depth of `point`.
    pub fn world_per_pixel(&self, point: Vec3, viewport_height: f32) -> f32 {
        let depth = (point - self.eye()).dot(self.forward()).max(1e-4);
        2.0 * depth * (self.fov_y * 0.5).tan() / viewport_height.max(1.0)
    }

    /// Project a world point to viewport pixels.
    pub fn project(&self, p: Vec3, size: Vec2) -> Option<Vec2> {
        let clip = self.proj(size.x / size.y) * self.view() * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new((ndc.x + 1.0) * 0.5 * size.x, (1.0 - ndc.y) * 0.5 * size.y))
    }

    pub fn orbit(&mut self, delta_px: Vec2) {
        self.yaw -= delta_px.x * 0.008;
        self.pitch = (self.pitch + delta_px.y * 0.008).clamp(-1.55, 1.55);
    }

    pub fn pan(&mut self, delta_px: Vec2, viewport_height: f32) {
        let k = self.world_per_pixel(self.target, viewport_height);
        self.target += (-self.right() * delta_px.x + self.up() * delta_px.y) * k;
    }

    pub fn dolly(&mut self, amount: f32) {
        self.distance = (self.distance * (1.0 - amount).clamp(0.2, 5.0)).max(1e-3);
    }
}
