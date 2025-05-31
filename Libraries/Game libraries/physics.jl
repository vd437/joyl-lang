// physics.joyl - Advanced Physics Simulation
pub struct PhysicsWorld {
    gravity: Vector3,
    collision_config: CollisionConfiguration,
    dispatcher: CollisionDispatcher,
    broadphase: DbvtBroadphase,
    solver: SequentialImpulseConstraintSolver,
    dynamics_world: DiscreteDynamicsWorld,
    rigid_bodies: [RigidBody],
    collision_shapes: [CollisionShape],
    debug_drawer: Option<PhysicsDebugDrawer>
}

impl PhysicsWorld {
    /// Create new physics world with specified gravity
    pub fn new(gravity: Vector3) -> PhysicsWorld {
        let config = DefaultCollisionConfiguration::new();
        let dispatcher = CollisionDispatcher::new(&config);
        let broadphase = DbvtBroadphase::new();
        let solver = SequentialImpulseConstraintSolver::new();
        
        let mut world = DiscreteDynamicsWorld::new(
            &dispatcher,
            &broadphase,
            &solver,
            &config
        );
        world.set_gravity(gravity);
        
        PhysicsWorld {
            gravity,
            collision_config: config,
            dispatcher,
            broadphase,
            solver,
            dynamics_world: world,
            rigid_bodies: [],
            collision_shapes: [],
            debug_drawer: None
        }
    }

    /// Add rigid body to simulation
    pub fn add_rigid_body(
        &mut self,
        shape: CollisionShape,
        mass: float,
        transform: Transform,
        is_kinematic: bool = false
    ) -> RigidBodyHandle {
        let motion_state = DefaultMotionState::new(transform);
        let local_inertia = shape.calculate_local_inertia(mass);
        
        let body = RigidBody::new(
            mass,
            motion_state,
            shape,
            local_inertia
        );
        
        let handle = self.dynamics_world.add_rigid_body(body);
        self.rigid_bodies.push(body);
        self.collision_shapes.push(shape);
        
        handle
    }

    /// Step the simulation forward
    pub fn step_simulation(
        &mut self,
        delta_time: float,
        max_sub_steps: int = 10,
        fixed_time_step: float = 1.0 / 60.0
    ) {
        self.dynamics_world.step_simulation(
            delta_time,
            max_sub_steps,
            fixed_time_step
        );
    }

    /// Raycast into the world
    pub fn ray_test(
        &self,
        from: Vector3,
        to: Vector3
    ) -> Option<RayResult> {
        let mut callback = ClosestRayResultCallback::new(from, to);
        self.dynamics_world.ray_test(from, to, &mut callback);
        
        if callback.has_hit() {
            Some(RayResult {
                hit_point: callback.hit_point_world(),
                hit_normal: callback.hit_normal_world(),
                rigid_body: callback.collision_object()
            })
        } else {
            None
        }
    }
}