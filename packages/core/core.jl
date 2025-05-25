// Joyl Core Library v1.0
// Single-file implementation
// Location: packages/joyl/core/core.jl

module joyl.core {
    // ==================== CORE TYPES ====================
    pub type Null;
    pub type Bool;
    pub type Int;
    pub type Float;
    pub type String;
    pub type Char;
    pub type Array<T>;
    pub type Map<K,V>;
    pub type Function;
    pub type Pointer<T>;
    pub type Result<T,E>; // Rust-like Result type for error handling

    // ==================== ERROR HANDLING ====================
    pub trait Error {
        fn message() -> String;
        fn cause() -> Option<Error>;
    }

    pub struct RuntimeError: Error {
        msg: String,
        cause: Option<Error>
    }

    impl RuntimeError {
        pub fn new(msg: String, cause: Option<Error>) -> Self {
            Self { msg, cause }
        }

        pub fn message(&self) -> String {
            self.msg.clone()
        }

        pub fn cause(&self) -> Option<Error> {
            self.cause.clone()
        }
    }

    // ==================== MEMORY MANAGEMENT ====================
    pub unsafe fn alloc<T>(size: Int) -> Pointer<T> {
        native unsafe_alloc(size)
    }

    pub unsafe fn free<T>(ptr: Pointer<T>) {
        native unsafe_free(ptr)
    }

    pub fn gc_enable(enable: Bool) {
        native gc_toggle(enable)
    }

    pub fn gc_collect() {
        native gc_run()
    }

    // ==================== CORE FUNCTIONS ====================
    pub fn print(value: Any) {
        native print_value(value)
    }

    pub fn println(value: Any) {
        native print_value(value)
        native print_newline()
    }

    pub fn panic(msg: String) -> ! {
        native panic(msg)
    }

    pub fn assert(condition: Bool, msg: String) {
        if !condition {
            panic(msg)
        }
    }

    // ==================== SYSTEM INTERACTION ====================
    pub fn get_env(key: String) -> Result<String, Error> {
        native sys_get_env(key)
    }

    pub fn set_env(key: String, value: String) -> Result<Null, Error> {
        native sys_set_env(key, value)
    }

    pub fn current_time() -> Int {
        native sys_current_time()
    }

    pub fn sleep(ms: Int) {
        native sys_sleep(ms)
    }

    // ==================== TYPE CONVERSION ====================
    pub fn to_int(value: Any) -> Result<Int, Error> {
        match value {
            Int(i) => Ok(i),
            Float(f) => Ok(f as Int),
            String(s) => {
                try {
                    Ok(s.parse<Int>())
                } catch (e) {
                    Err(RuntimeError::new("Failed to parse string to int", Some(e)))
                }
            },
            _ => Err(RuntimeError::new("Cannot convert type to Int", None))
        }
    }

    pub fn to_float(value: Any) -> Result<Float, Error> {
        match value {
            Float(f) => Ok(f),
            Int(i) => Ok(i as Float),
            String(s) => {
                try {
                    Ok(s.parse<Float>())
                } catch (e) {
                    Err(RuntimeError::new("Failed to parse string to float", Some(e)))
                }
            },
            _ => Err(RuntimeError::new("Cannot convert type to Float", None))
        }
    }

    pub fn to_string(value: Any) -> String {
        native value_to_string(value)
    }

    pub fn to_bool(value: Any) -> Bool {
        native value_to_bool(value)
    }

    // ==================== UTILITY FUNCTIONS ====================
    pub fn type_of(value: Any) -> String {
        native get_type_name(value)
    }

    pub fn is_null(value: Any) -> Bool {
        match value {
            Null => true,
            _ => false
        }
    }

    pub fn clone<T>(value: T) -> T {
        native deep_clone(value)
    }

    pub fn hash<T>(value: T) -> Int {
        native compute_hash(value)
    }

    // ==================== ITERATION PROTOCOLS ====================
    pub trait Iterable<T> {
        fn iter() -> Iterator<T>;
    }

    pub trait Iterator<T> {
        fn next() -> Option<T>;
    }

    // ==================== OPTIONAL VALUES ====================
    pub enum Option<T> {
        Some(T),
        None
    }

    impl<T> Option<T> {
        pub fn unwrap(self) -> T {
            match self {
                Some(val) => val,
                None => panic("Called unwrap on None value")
            }
        }

        pub fn unwrap_or(self, default: T) -> T {
            match self {
                Some(val) => val,
                None => default
            }
        }
    }
}
