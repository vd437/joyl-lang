Getting Started with Joyl

Welcome to Joyl — a modern programming language that blends Python simplicity with C and Rust-level performance. This guide will help you get up and running in minutes.


---

1. Install Joyl (Development Setup)

First, clone the repository and install Joyl in editable mode:

git clone https://github.com/vd437/joyl.git
cd joyl
python -m pip install -e .

Requirements:

Python 3.10 or higher

Git



---

2. Run Your First Joyl Program

You can run .jl files using the Joyl interpreter:

python src/joyl.py examples/hello_world.jl

Make sure the interpreter is located in src/joyl.py and the examples are in the examples/ folder.


---

3. Hello World Example

Create a file named hello.jl with the following code:

print("Hello from Joyl!")

Then run it:

python src/joyl.py hello.jl


---

4. Explore Built-in Examples

You can test and learn from ready-made examples:

cd examples/
python ../src/joyl.py calculator.jl

Examples include:

calculator.jl

classes.jl

loops.jl

error_handling.jl



---

5. Use the Debug Mode (Optional)

Debug your program to see execution steps:

python src/joyl.py --debug examples/yourfile.jl


---

6. Recommended Tools

Install these tools for a better experience:

pytest for running tests

black for code formatting

VS Code Extensions:

Python

Pylance

Mermaid Preview--

7. What’s Next?

Learn the Syntax

Understand the Architecture

See our Changelog

Want to contribute? Read the Contribution Guide



---

Joyl is an open-source project. Join us and help shape the future of programming!
