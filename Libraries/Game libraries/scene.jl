// scene.joyl - Hierarchical Scene Management
pub struct SceneGraph {
    root: SceneNode,
    node_registry: HashMap<Uuid, SceneNode>,
    systems: SceneSystems,
    active_camera: Option<CameraNode>,
    render_queue: RenderQueue
}

impl SceneGraph {
    /// Create empty scene
    pub fn new() -> SceneGraph {
        SceneGraph {
            root: SceneNode::new("Root"),
            node_registry: HashMap::new(),
            systems: SceneSystems::new(),
            active_camera: None,
            render_queue: RenderQueue::new()
        }
    }

    /// Add entity to scene
    pub fn add_entity(
        &mut self,
        name: string,
        components: [Component],
        parent: Option<Uuid>
    ) -> Uuid {
        let node = SceneNode::new(name);
        let id = node.id;
        
        // Add components
        for component in components {
            node.add_component(component);
            
            // Special handling for camera
            if let Component::Camera(camera) = component {
                self.active_camera = Some(CameraNode {
                    node_id: id,
                    camera
                });
            }
        }
        
        // Add to hierarchy
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.node_registry.get_mut(&parent_id) {
                parent_node.add_child(node);
            }
        } else {
            self.root.add_child(node);
        }
        
        self.node_registry.insert(id, node);
        id
    }

    /// Update all scene systems
    pub fn update(&mut self, delta_time: float) {
        // Update transforms
        self.systems.transform_system.update(
            &mut self.root,
            delta_time
        );
        
        // Process animations
        self.systems.animation_system.update(
            &mut self.node_registry,
            delta_time
        );
        
        // Build render queue
        self.systems.rendering_system.process(
            &self.root,
            &mut self.render_queue
        );
    }

    /// Get renderable objects
    pub fn get_render_queue(&self) -> &RenderQueue {
        &self.render_queue
    }
}