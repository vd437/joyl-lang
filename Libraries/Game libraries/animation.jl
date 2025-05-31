// animation.joyl - Skeletal Animation
pub struct AnimationSystem {
    animations: HashMap<String, AnimationClip>,
    skeletons: HashMap<Uuid, Skeleton>,
    current_poses: HashMap<Uuid, Pose>,
    blending: AnimationBlendingSystem,
    ik_system: InverseKinematicsSystem
}

impl AnimationSystem {
    /// Load animation from file
    pub fn load_animation(
        &mut self,
        path: string,
        name: string
    ) -> Result<(), AnimationError> {
        let anim_data = AnimationLoader::load(path)?;
        self.animations.insert(name, anim_data);
        Ok(())
    }

    /// Play animation on entity
    pub fn play_animation(
        &mut self,
        entity_id: Uuid,
        anim_name: string,
        blend_time: float = 0.2,
        looped: bool = true
    ) {
        if let Some(anim) = self.animations.get(&anim_name) {
            if let Some(skeleton) = self.skeletons.get(&entity_id) {
                self.blending.start_blend(
                    entity_id,
                    anim,
                    skeleton,
                    blend_time,
                    looped
                );
            }
        }
    }

    /// Update all animations
    pub fn update(&mut self, delta_time: float) {
        // Update animation states
        for (entity_id, blend) in &mut self.blending.active_blends {
            blend.update(delta_time);
            
            // Apply to skeleton
            if let Some(skeleton) = self.skeletons.get(entity_id) {
                let pose = blend.current_pose();
                self.current_poses.insert(*entity_id, pose);
                
                // Solve IK
                self.ik_system.solve(skeleton, &mut pose);
            }
        }
    }
}