# Joyl Standard Library Reference

## Core Modules

### 1. `io` - Input/Output
```joyl
// File operations
let file = io.open("data.txt", "r")
let content = file.read()
file.close()

// Standard I/O
io.print("Hello")  // Alias for print()
let input = io.input("Enter value: ")
```

### 2. `math` - Mathematical Operations
```joyl
math.pi       // 3.141592653589793
math.sin(0.5) // Sine function
math.log(10)  // Natural logarithm

// Constants
math.E        // Euler's number
math.PHI      // Golden ratio

// Functions
math.floor(3.7)  // 3
math.round(3.7)  // 4
math.clamp(10, 0, 5)  // 5
```

### 3. `collections` - Data Structures
```joyl
// Array utilities
let nums = [1, 2, 3]
collections.push(nums, 4)  // [1,2,3,4]
collections.reverse(nums)  // [4,3,2,1]

// Dictionary operations
let dict = {"a": 1, "b": 2}
collections.keys(dict)    // ["a", "b"]
collections.merge(dict, {"c": 3})
```

### 4. `strings` - String Manipulation
```joyl
strings.trim("  hello  ")  // "hello"
strings.split("a,b,c", ",")  // ["a","b","c"]
strings.format("{} + {} = {}", 2, 3, 5)  // "2 + 3 = 5"
```

### 5. `datetime` - Date/Time Handling
```joyl
let now = datetime.now()
datetime.format(now, "YYYY-MM-DD")  // "2023-11-15"

let tomorrow = datetime.add_days(now, 1)
datetime.diff(now, tomorrow)  // 86400 (seconds)
```

## Concurrency Modules

### 6. `threading` - Basic Threads
```joyl
fn task() {
    print("Running in thread")
}

let t = threading.new_thread(task)
t.start()
t.join()
```

### 7. `async` - Asynchronous Operations
```joyl
async fn fetch_data(url) {
    let response = await http.get(url)
    return response
}
```

## Specialized Modules

### 8. `json` - JSON Handling
```joyl
let data = json.parse('{"name":"John"}')  // {name:"John"}
let str = json.stringify(data)  // '{"name":"John"}'
```

### 9. `random` - Random Generation
```joyl
random.int(1, 10)  // Random integer
random.choice(["a","b","c"])  // Random element
```

### 10. `regex` - Regular Expressions
```joyl
let pattern = regex.compile(r"\d+")
regex.match(pattern, "123")  // true
```

## Utility Modules

### 11. `system` - System Interaction
```joyl
system.os()       // "linux", "windows", etc.
system.time()     // Current timestamp
system.exit(0)    // Terminate program
```

### 12. `debug` - Debugging Tools
```joyl
debug.log("Variable value:", x)  // Logs with timestamp
debug.trace()       // Print call stack
```

## Example Usage
```joyl
import math, datetime

fn calculate_circle(r) {
    return math.pi * r ** 2
}

print("Today is", datetime.format(datetime.now(), "YYYY-MM-DD"))
print("Area:", calculate_circle(5))
```

## Version Compatibility
| Module       | Since Version |
|--------------|--------------|
| Core Modules | 1.0          |
| Async        | 1.2          |
| JSON         | 1.1          |
```

This documentation:
1. Organizes modules by category
2. Shows practical examples for each
3. Includes version information
4. Uses consistent Joyl syntax
5. Provides both basic and advanced functionality