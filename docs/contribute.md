# Contributing to Joyl Language

Welcome to the Joyl programming language! This document will guide you on how to contribute effectively to the project and join our growing community.

---

## 🚀 First Time Contributors
We welcome all contributors! To get started:

1. Fork the repository
2. Clone it to your machine
3. Look for issues labeled `good first issue`
4. Make your changes and submit a pull request

```bash
git clone https://github.com/vd43i/joyl.git
cd joyl
python -m pip install -e .


---

💻 Development Setup

Prerequisites

Python 3.10+

pytest for testing

black for code formatting


Recommended VS Code Extensions

Python

Pylance

Mermaid Preview (for architecture diagrams)



---

🔍 Contribution Areas

Area	Skills Needed	Starter Issues

Compiler	Parsing, Bytecode	#42, #15
Standard Lib	Algorithms, APIs	#33
Documentation	Technical Writing	#21
Tooling	CLI, Developer UX	#18



---

🛠 Coding Standards

1. Use snake_case for Python code


2. Use camelCase for Joyl internals


3. Always add type hints:



def lex(source: str) -> List[Token]:

4. Use black to format your code:



black src/


---

📝 Pull Request Checklist

Before submitting a pull request, ensure the following:

[ ] All tests pass (pytest)

[ ] Documentation is updated

[ ] Changelog entry is added if needed

[ ] Code is rebased on the latest main branch



---

🐛 Debugging Tips

Run the interpreter with debug mode:

python src/joyl.py --debug examples/yourfile.jl


---

🏆 Contributor Recognition

Top contributors receive:

Mention in release notes

Special Discord role

Custom GitHub profile badge



---

💬 Join Our Community

Connect with the Joyl team:

Discord (real-time help & chat)

GitHub Discussions (proposals, ideas, Q&A)



---

📘 See Also

Project README: Overview, examples, and goals of Joyl




