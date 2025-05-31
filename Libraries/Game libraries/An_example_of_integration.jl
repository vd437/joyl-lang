import { GraphicsEngine, PhysicsWorld, SceneGraph } from "joyl_game";

fn main() {
    // Initialize systems
    let mut graphics = GraphicsEngine::new("My Game", 1280, 720);
    let mut physics = PhysicsWorld::new([0.0, -9.81, 0.0]);
    let mut scene = SceneGraph::new();
    
    // Load assets
    graphics.load_shader("basic", "shaders/basic.vert", "shaders/basic.frag");
    
    // Create entities
    let player_id = scene.add_entity("Player", [
        Component::Transform(Transform::identity()),
        Component::Model("models/player.glb"),
        Component::RigidBody(physics.create_rigidbody(...))
    ], None);
    
    // Main game loop
    while !graphics.window_should_close() {
        let delta = graphics.get_delta_time();
        
        // Update systems
        physics.step_simulation(delta);
        scene.update(delta);
        
        // Render frame
        graphics.begin_frame();
        graphics.render_scene(&scene);
        graphics.end_frame();
    }
}