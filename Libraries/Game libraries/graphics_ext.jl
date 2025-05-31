// graphics_ext.joyl - Advanced Rendering Features
pub struct DeferredRenderer {
    g_buffer: GBuffer,
    lighting_pass: ShaderProgram,
    shadow_mapping: ShadowSystem,
    post_processing: PostProcessPipeline,
    ssao: SSAOSystem,
    volumetric: VolumetricLighting
}

impl DeferredRenderer {
    /// Initialize deferred renderer
    pub fn new(
        width: uint,
        height: uint,
        settings: RenderSettings
    ) -> DeferredRenderer {
        DeferredRenderer {
            g_buffer: GBuffer::new(width, height),
            lighting_pass: ShaderManager::load(
                "deferred_lighting",
                "shaders/lighting.vert",
                "shaders/lighting.frag"
            ),
            shadow_mapping: ShadowSystem::new(
                settings.shadow_resolution,
                settings.cascade_count
            ),
            post_processing: PostProcessPipeline::new(width, height),
            ssao: SSAOSystem::new(width, height),
            volumetric: VolumetricLighting::new()
        }
    }

    /// Render full frame
    pub fn render(
        &mut self,
        scene: &SceneGraph,
        camera: &Camera
    ) -> FrameBuffer {
        // Geometry pass
        self.g_buffer.begin_geometry_pass();
        for renderable in scene.get_render_queue() {
            self.process_geometry(renderable);
        }
        self.g_buffer.end_geometry_pass();
        
        // Shadow pass
        self.shadow_mapping.render_shadows(scene);
        
        // Lighting pass
        self.lighting_pass.bind();
        self.apply_lighting(scene.lights());
        
        // Post-processing
        self.post_processing.apply_effects();
        
        self.post_processing.get_output()
    }
}