# Joyl Language - Syntax Reference

## 1. Variables and Constants

```joyl
// Variable declaration
let name = "John"   // String
let age = 30        // Integer
let price = 9.99    // Float
let active = true   // Boolean

// Constants (immutable)
const PI = 3.14159
const COMPANY = "Joyl"
```

## 2. Basic Data Types

```joyl
// Numbers
let integerNum = 42     // Integer
let floatNum = 3.14     // Float

// Strings
let greeting = "Hello World!"
let multiLine = """
  This is a multi-line
  string
"""

// Booleans
let isTrue = true
let isFalse = false

// Null value
let nothing = nil
```

## 3. Operators

```joyl
// Arithmetic operators
let sum = 10 + 5    // 15
let sub = 10 - 5    // 5
let mul = 10 * 5    // 50
let div = 10 / 5    // 2
let mod = 10 % 3    // 1

// Logical operators
let and = true && false  // false
let or = true || false   // true
let not = !true          // false

// Comparison operators
let eq = 5 == 5     // true
let neq = 5 != 3    // true
let gt = 5 > 3      // true
let lt = 5 < 3      // false
```

## 4. Control Flow

### Conditional Statements

```joyl
let age = 18

if age >= 18 {
    print("You can vote")
} else {
    print("You cannot vote yet")
}

// Ternary-style
let status = if age >= 18 { "Adult" } else { "Minor" }
```

### Loops

```joyl
// For loop
for i in 1..5 {
    print(i)  // Prints 1, 2, 3, 4
}

// While loop
let j = 0
while j < 3 {
    print(j)
    j += 1
}
```

## 5. Functions

```joyl
// Basic function
fn greet(name) {
    print("Hello " + name)
}

// Function with parameters and return
fn add(a, b) {
    return a + b
}

// Function calls
greet("John")       // Prints: Hello John
let result = add(2, 3)  // Returns 5

// Lambda functions
const square = fn(x) { x * x }
square(5)  // 25
```

## 6. Arrays/Lists

```joyl
// Create array
let fruits = ["Apple", "Banana", "Orange"]

// Access elements
let first = fruits[0]  // "Apple"

// Add element
fruits.push("Strawberry")

// Array length
let count = fruits.len()  // 4

// Iterate through array
for fruit in fruits {
    print(fruit)
}
```

## 7. Objects/Dictionaries

```joyl
// Create object
let person = {
    "name": "John",
    "age": 30,
    "isStudent": true
}

// Access properties
print(person["name"])  // "John"

// Add new property
person["job"] = "Developer"
```

## 8. Error Handling

```joyl
try {
    // Code that might fail
    let result = riskyOperation()
} catch err {
    // Handle error
    print("Error occurred: " + err)
} finally {
# Joyl Language - Syntax Reference

## 1. Variables and Constants

```joyl
// Variable declaration
let name = "John"   // String
let age = 30        // Integer
let price = 9.99    // Float
let active = true   // Boolean

// Constants (immutable)
const PI = 3.14159
const COMPANY = "Joyl"
```

## 2. Basic Data Types

```joyl
// Numbers
let integerNum = 42     // Integer
let floatNum = 3.14     // Float

// Strings
let greeting = "Hello World!"
let multiLine = """
  This is a multi-line
  string
"""

// Booleans
let isTrue = true
let isFalse = false

// Null value
let nothing = nil
```

## 3. Operators

```joyl
// Arithmetic operators
let sum = 10 + 5    // 15
let sub = 10 - 5    // 5
let mul = 10 * 5    // 50
let div = 10 / 5    // 2
let mod = 10 % 3    // 1

// Logical operators
let and = true && false  // false
let or = true || false   // true
let not = !true          // false

// Comparison operators
let eq = 5 == 5     // true
let neq = 5 != 3    // true
let gt = 5 > 3      // true
let lt = 5 < 3      // false
```

## 4. Control Flow

### Conditional Statements

```joyl
let age = 18

if age >= 18 {
    print("You can vote")
} else {
    print("You cannot vote yet")
}

// Ternary-style
let status = if age >= 18 { "Adult" } else { "Minor" }
```

### Loops

```joyl
// For loop
for i in 1..5 {
    print(i)  // Prints 1, 2, 3, 4
}

// While loop
let j = 0
while j < 3 {
    print(j)
    j += 1
}
```

## 5. Functions

```joyl
// Basic function
fn greet(name) {
    print("Hello " + name)
}

// Function with parameters and return
fn add(a, b) {
    return a + b
}

// Function calls
greet("John")       // Prints: Hello John
let result = add(2, 3)  // Returns 5

// Lambda functions
const square = fn(x) { x * x }
square(5)  // 25
```

## 6. Arrays/Lists

```joyl
// Create array
let fruits = ["Apple", "Banana", "Orange"]

// Access elements
let first = fruits[0]  // "Apple"

// Add element
fruits.push("Strawberry")

// Array length
let count = fruits.len()  // 4

// Iterate through array
for fruit in fruits {
    print(fruit)
}
```

## 7. Objects/Dictionaries

```joyl
// Create object
let person = {
    "name": "John",
    "age": 30,
    "isStudent": true
}

// Access properties
print(person["name"])  // "John"

// Add new property
person["job"] = "Developer"
```

## 8. Error Handling

```joyl
try {
    // Code that might fail
    let result = riskyOperation()
} catch err {
    // Handle error
    print("Error occurred: " + err)
} finally {
    // Always executes
    print("Operation completed")
}
```

## 9. Modules and Imports

```joyl
// Import entire module
import "math"

// Import specific functions
import { sqrt, pow } from "math"

// Import with alias
import "math" as m
```

## 10. Practical Examples

### Example 1: Simple Calculator

```joyl
fn calculate(a, b, op) {
    return match op {
        "+" => a + b
        "-" => a - b
        "*" => a * b
        "/" => a / b
        _ => nil  // Default case
    }
}

let result = calculate(10, 5, "+")  // 15
```

### Example 2: Student Management

```joyl
class Student {
    let name
    let grades = []
    
    fn new(name) {
        self.name = name
    }
    
    fn addGrade(grade) {
        self.grades.push(grade)
    }
    
    fn average() {
        if self.grades.empty() return 0
        
        let sum = 0
        for grade in self.grades {
            sum += grade
        }
        return sum / self.grades.len()
    }
}

let student = Student.new("John")
student.addGrade(95)
student.addGrade(87)
print(student.average())  // 91
```

## Next Steps
- Try these examples in the Joyl compiler
- Read advanced documentation for more features
- Contribute to the language development with your ideas￼Enter    // Always executes
    print("Operation completed")
}
```

## 9. Modules and Imports

```joyl
// Import entire module
import "math"

// Import specific functions
import { sqrt, pow } from "math"

// Import with alias
import "math" as m
```

## 10. Practical Examples

### Example 1: Simple Calculator

```joyl
fn calculate(a, b, op) {
    return match op {
        "+" => a + b
        "-" => a - b
  "*" => a * b
        "/" => a / b
        _ => nil  // Default case
    }
}

let result = calculate(10, 5, "+")  // 15
```

### Example 2: Student Management

```joyl
class Student {
    let name
    let grades = []
    
    fn new(name) {
        self.name = name
    }
    
    fn addGrade(grade) {
        self.grades.push(grade)
    }
    
    fn average() {
        if self.grades.empty() return 0
        
        let sum = 0
        for grade in self.grades {
            sum += grade
        }
        return sum / self.grades.len()
    }
}

let student = Student.new("John")
student.addGrade(95)
student.addGrade(87)
print(student.average())  // 91
```

## Next Steps
- Try these examples in the Joyl compiler
- Read advanced documentation for more features
- Contribute to the language development with your ideas
