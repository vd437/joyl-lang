// input.joyl - Advanced Input System
pub struct InputManager {
    keyboard_state: KeyboardState,
    mouse_state: MouseState,
    gamepad_manager: GamepadManager,
    event_queue: Vec<InputEvent>,
    input_mappings: InputMappingSystem,
    gesture_recognizer: GestureRecognizer
}

impl InputManager {
    /// Initialize input system
    pub fn new(window: &Window) -> InputManager {
        InputManager {
            keyboard_state: KeyboardState::new(),
            mouse_state: MouseState::new(window),
            gamepad_manager: GamepadManager::new(),
            event_queue: Vec::with_capacity(128),
            input_mappings: InputMappingSystem::new(),
            gesture_recognizer: GestureRecognizer::new()
        }
    }

    /// Process all input events
    pub fn update(&mut self) {
        self.keyboard_state.update();
        self.mouse_state.update();
        self.gamepad_manager.update();
        
        self.process_gestures();
        self.process_mappings();
    }

    /// Add input mapping
    pub fn add_mapping(
        &mut self,
        name: string,
        mapping: InputMapping
    ) -> Result<(), InputError> {
        self.input_mappings.add(name, mapping)
    }

    /// Check if action is triggered
    pub fn get_action(&self, name: string) -> bool {
        self.input_mappings.get_action(name)
    }
}