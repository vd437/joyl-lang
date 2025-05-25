# Getting Started with Joyl

Welcome to **Joyl** — a modern programming language that combines Python’s simplicity with the performance and memory safety of C and Rust.

This guide will help you set up Joyl and run your first program in just a few minutes.

---

## 1. Installation

### Requirements
- Python 3.10+
- Git

### Steps

```bash
git clone https://github.com/vd437/joyl.git
cd joyl
python -m pip install -e .


---

2. Running Your First Joyl Program

To run any .jl file using the Joyl interpreter:

python src/joyl.py examples/hello_world.jl


---

3. Writing a Hello World

Create a file named hello.jl with this code:

print("Hello from Joyl!")

Then run it:

python src/joyl.py hello.jl


---

4. Explore Built-in Examples

Joyl includes several examples in the examples/ directory:

Example Files

hello_world.jl — Print a message

calculator.jl — Basic calculator

loops.jl — Loop structures

classes.jl — Object-oriented features

error_handling.jl — Try/catch demonstration


Run an example:

python src/joyl.py examples/calculator.jl


---

5. Debug Mode

Use debug mode to trace each step of your program:

python src/joyl.py --debug examples/yourfile.jl


---

6. Developer Tools

Useful Python Tools

pytest — Run test cases

black — Format your code


VS Code Extensions

Python

Pylance

Mermaid Preview (for diagrams in documentation)



---

7. Next Steps

Learn more by exploring the rest of our documentation:

Syntax Guide

Architecture Overview

Contributing Guide

Changelog



---

8. Community & Contributions

Joyl is an open-source project maintained by passionate developers.

If you'd like to contribute, start with the Contribution Guide, and don't forget to join our discussions and Discord (coming soon).


---

Happy coding with Joyl!
