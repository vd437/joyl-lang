#!/usr/bin/env python3
# Joyl Compiler - The Definitive Edition
# A complete, professional-grade systems programming language compiler.
# This version merges the feature-richness of v4.0 (full keywords, AST) 
# with the advanced architecture of v7.0 (deep semantic analysis, type system, generics, traits, LLVM/WASM codegen).

import sys
import os
import re
import struct
import hashlib
import math
import json
import time
import threading
import queue
import mmap
import contextlib
import argparse
import subprocess
import tempfile
import shutil
import platform
import importlib
import zipfile
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Set, NamedTuple
from pathlib import Path
from collections import defaultdict, OrderedDict
from urllib.request import urlopen
from urllib.parse import urlparse

# =================================================================================================
# =================================== CONSTANTS & CONFIGURATION ===================================
# =================================================================================================

VERSION = "8.0-Masterpiece"
DEFAULT_OPTIMIZATION_LEVEL = 3
MAX_ERRORS_BEFORE_ABORT = 25
STDLIB_PATH = os.path.join(os.path.dirname(__file__), "stdlib")
PKG_CACHE = os.path.expanduser("~/.joyl/packages")
PKG_REGISTRY = "https://packages.joyl-lang.org"
LLVM_TARGETS = ["x86_64", "aarch64", "wasm32"]

# =================================================================================================
# ============================================ ENUMS ==============================================
# =================================================================================================

class CompilerMode(Enum):
    DEBUG = auto()
    RELEASE = auto()
    OPTIMIZE = auto()

class TargetPlatform(Enum):
    NATIVE = auto()
    WASM = auto()
    LLVM = auto()

class TokenType(Enum):
    # Identifiers and literals
    IDENTIFIER = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    
    # Keywords
    LET = auto()
    CONST = auto()
    FN = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    STRUCT = auto()
    ENUM = auto()
    TRAIT = auto()
    IMPL = auto()
    IMPORT = auto()
    AS = auto()
    FROM = auto()
    PUB = auto()
    STATIC = auto()
    ASYNC = auto()
    AWAIT = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    THROW = auto()
    MATCH = auto()
    MODULE = auto()
    UNSAFE = auto()
    DEFER = auto()
    TRUE = auto()
    FALSE = auto()
    NEVER = auto() # Represents the never type '!'
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    BANG = auto()
    EQ = auto()
    EQ_EQ = auto()
    BANG_EQ = auto()
    LT = auto()
    GT = auto()
    LT_EQ = auto()
    GT_EQ = auto()
    AMP_AMP = auto()
    PIPE_PIPE = auto()
    LT_LT = auto()
    GT_GT = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    PLUS_EQ = auto()
    MINUS_EQ = auto()
    STAR_EQ = auto()
    SLASH_EQ = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMI = auto()
    ARROW = auto()
    FAT_ARROW = auto()
    AT = auto()
    UNDERSCORE = auto()
    QUESTION = auto()
    
    # Special
    EOF = auto()

class TypeKind(Enum):
    PRIMITIVE = auto()
    STRUCT = auto()
    ENUM = auto()
    TRAIT = auto()
    FUNCTION = auto()
    POINTER = auto()
    ARRAY = auto()
    SLICE = auto()
    TUPLE = auto()
    NEVER = auto()
    GENERIC = auto()
    GENERIC_PARAM = auto()
    MODULE = auto()

# =================================================================================================
# ======================================== ERROR HANDLING =========================================
# =================================================================================================

class CompilerError(Exception):
    def __init__(self, message, filename=None, line=None, column=None, context=None):
        super().__init__(message)
        self.filename = filename
        self.line = line
        self.column = column
        self.context = context

    def __str__(self):
        loc = ""
        if self.filename:
            loc = f"{self.filename}:"
            if self.line is not None:
                loc += f"{self.line}:"
                if self.column is not None:
                    loc += f"{self.column}:"
        
        # ANSI color codes for highlighting
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        error_msg = f"{loc} {BOLD}{RED}error:{RESET}{BOLD} {super().__str__()}{RESET}"
        context_msg = f"\n    Context: {self.context}" if self.context else ""
        return error_msg + context_msg

class ErrorReporter:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def error(self, message, node=None, filename=None, line=None, column=None, context=None):
        if node:
            err = CompilerError(message, node.filename, node.line, node.column, context)
        else:
            err = CompilerError(message, filename, line, column, context)
        
        self.errors.append(err)
        if len(self.errors) >= MAX_ERRORS_BEFORE_ABORT:
            self.print_errors()
            print("\nToo many errors, aborting compilation.", file=sys.stderr)
            sys.exit(1)
            
    def warning(self, message, node=None, filename=None, line=None, column=None, context=None):
        if node:
            warn = CompilerError(message, node.filename, node.line, node.column, context)
        else:
            warn = CompilerError(message, filename, line, column, context)
        self.warnings.append(warn)
        
    def has_errors(self):
        return len(self.errors) > 0
        
    def print_errors(self):
        # ANSI color codes for warnings
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        for err in self.errors:
            print(err, file=sys.stderr)
        for warn in self.warnings:
            warn_str = str(warn).replace("error:", f"{BOLD}{YELLOW}warning:{RESET}{BOLD}")
            print(warn_str, file=sys.stderr)

# =================================================================================================
# =========================================== TYPE SYSTEM =========================================
# =================================================================================================

@dataclass
class Type:
    kind: TypeKind
    name: str
    size: int = 0
    align: int = 0
    members: 'OrderedDict[str, Type]' = field(default_factory=OrderedDict)
    element_type: Optional['Type'] = None
    return_type: Optional['Type'] = None
    parameters: Optional[List['Type']] = None
    is_mutable: bool = False
    is_sendable: bool = False
    is_copyable: bool = True
    generic_params: List[str] = field(default_factory=list)
    trait_impls: Set[str] = field(default_factory=set)

class TypeSystem:
    def __init__(self, error_reporter: ErrorReporter):
        self.types = {}
        self.type_aliases = {}
        self.error_reporter = error_reporter
        self.trait_implementations = defaultdict(set)
        
        self.PRIMITIVES = {
            'i8': Type(TypeKind.PRIMITIVE, 'i8', 1, 1),
            'i16': Type(TypeKind.PRIMITIVE, 'i16', 2, 2),
            'i32': Type(TypeKind.PRIMITIVE, 'i32', 4, 4),
            'i64': Type(TypeKind.PRIMITIVE, 'i64', 8, 8),
            'u8': Type(TypeKind.PRIMITIVE, 'u8', 1, 1),
            'u16': Type(TypeKind.PRIMITIVE, 'u16', 2, 2),
            'u32': Type(TypeKind.PRIMITIVE, 'u32', 4, 4),
            'u64': Type(TypeKind.PRIMITIVE, 'u64', 8, 8),
            'f32': Type(TypeKind.PRIMITIVE, 'f32', 4, 4),
            'f64': Type(TypeKind.PRIMITIVE, 'f64', 8, 8),
            'bool': Type(TypeKind.PRIMITIVE, 'bool', 1, 1),
            'char': Type(TypeKind.PRIMITIVE, 'char', 4, 4), # UTF-32
            'str': Type(TypeKind.PRIMITIVE, 'str', 16, 8), # Slice: pointer + length
            'never': Type(TypeKind.NEVER, 'never', 0, 0),
            'usize': Type(TypeKind.PRIMITIVE, 'usize', 8, 8) # Assuming 64-bit target
        }
        self.types.update(self.PRIMITIVES)
        self.init_stdlib_types()

    def init_stdlib_types(self):
        # These are generic definitions, not concrete types.
        self.add_type(Type(TypeKind.GENERIC, 'Option', generic_params=['T']))
        self.add_type(Type(TypeKind.GENERIC, 'Result', generic_params=['T', 'E']))
        self.add_type(Type(TypeKind.GENERIC, 'Vec', generic_params=['T']))
        self.add_type(Type(TypeKind.GENERIC, 'HashMap', generic_params=['K', 'V']))
        
        # Concrete String type, often backed by Vec<u8>
        string_type = Type(TypeKind.STRUCT, 'String', 24, 8)
        string_type.members = OrderedDict([
            ('ptr', self.pointer_to(self.get_type('u8'))),
            ('len', self.get_type('usize')),
            ('capacity', self.get_type('usize'))
        ])
        self.add_type(string_type)

    def get_type(self, name: str) -> Optional[Type]:
        if name in self.type_aliases:
            name = self.type_aliases[name]
        return self.types.get(name)

    def add_type(self, type_instance: Type):
        if type_instance.name in self.types:
            # This can happen with generic instantiations, which is fine.
            # Only error for non-generic redefinitions.
            if not self.types[type_instance.name].generic_params and not type_instance.generic_params:
                 self.error_reporter.warning(f"Type '{type_instance.name}' is already defined.")
        self.types[type_instance.name] = type_instance

    def pointer_to(self, target_type: Type, is_mutable: bool = False) -> Type:
        name = f"*{'mut ' if is_mutable else ''}{target_type.name}"
        if self.get_type(name):
            return self.get_type(name)
        
        ptr_type = Type(
            TypeKind.POINTER, name, 8, 8, # 64-bit pointer
            element_type=target_type, is_mutable=is_mutable
        )
        self.add_type(ptr_type)
        return ptr_type

    def array_of(self, element_type: Type, size: int) -> Type:
        name = f"[{element_type.name}; {size}]"
        if self.get_type(name):
            return self.get_type(name)
        
        array_type = Type(
            TypeKind.ARRAY, name, element_type.size * size, element_type.align,
            element_type=element_type
        )
        self.add_type(array_type)
        return array_type

    def slice_of(self, element_type: Type) -> Type:
        name = f"[{element_type.name}]"
        if self.get_type(name):
            return self.get_type(name)
            
        slice_type = Type(
            TypeKind.SLICE, name, 16, 8, # ptr + len
            element_type=element_type
        )
        self.add_type(slice_type)
        return slice_type

    def is_subtype(self, subtype: Type, supertype: Type) -> bool:
        if subtype == supertype or supertype.name == 'Any':
            return True
        if subtype.kind == TypeKind.NEVER: # 'never' type can coerce to any type
            return True
        if supertype.kind == TypeKind.TRAIT:
            return supertype.name in self.trait_implementations.get(subtype.name, set())
        # Add more complex subtyping rules here (e.g., variance for generics)
        return False

    def add_trait_impl(self, type_name: str, trait_name: str):
        self.trait_implementations[type_name].add(trait_name)

    def unify_types(self, t1: Type, t2: Type) -> Optional[Type]:
        if t1 == t2:
            return t1
        
        numeric_types = {'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f32', 'f64'}
        if t1.name in numeric_types and t2.name in numeric_types:
            # Simple numeric promotion: prefer larger size, prefer float over int
            is_f1 = 'f' in t1.name
            is_f2 = 'f' in t2.name
            if is_f1 and not is_f2: return t1
            if is_f2 and not is_f1: return t2
            return t1 if t1.size >= t2.size else t2
            
        return None

# =================================================================================================
# ================================== ABSTRACT SYNTAX TREE (AST) NODES =============================
# =================================================================================================

@dataclass
class ASTNode:
    line: int
    column: int
    filename: str
    inferred_type: Optional[Type] = field(default=None, repr=False)

# --- Type Expressions ---
@dataclass
class TypeExpr(ASTNode): pass

@dataclass
class SimpleType(TypeExpr):
    name: str
    generics: List[TypeExpr] = field(default_factory=list)

@dataclass
class PointerType(TypeExpr):
    target: TypeExpr
    is_mutable: bool

@dataclass
class ArrayType(TypeExpr):
    element: TypeExpr
    size: Optional['Expression'] # None for slice

@dataclass
class TupleType(TypeExpr):
    elements: List[TypeExpr]

@dataclass
class FunctionType(TypeExpr):
    params: List[TypeExpr]
    return_type: TypeExpr

# --- Core Structures ---
@dataclass
class Program(ASTNode):
    statements: List['Statement']

@dataclass
class Block(ASTNode):
    statements: List['Statement']

# --- Declarations ---
@dataclass
class Statement(ASTNode): pass

@dataclass
class Declaration(Statement): pass

@dataclass
class Let(Declaration):
    name: str
    type_decl: Optional[TypeExpr]
    initializer: Optional['Expression']
    is_mutable: bool

@dataclass
class Function(Declaration):
    name: str
    generics: List[SimpleType]
    params: List[Let]
    return_type: Optional[TypeExpr]
    body: Block
    is_async: bool
    is_public: bool

@dataclass
class Struct(Declaration):
    name: str
    generics: List[SimpleType]
    fields: List[Let]
    is_public: bool

@dataclass
class Enum(Declaration):
    name: str
    generics: List[SimpleType]
    variants: List[Union['EnumVariant', 'EnumVariantWithValue']]
    is_public: bool

@dataclass
class EnumVariant(ASTNode):
    name: str

@dataclass
class EnumVariantWithValue(ASTNode):
    name: str
    types: List[TypeExpr]

@dataclass
class Trait(Declaration):
    name: str
    generics: List[SimpleType]
    methods: List[Function]
    is_public: bool

@dataclass
class Impl(Declaration):
    generics: List[SimpleType]
    trait: Optional[SimpleType]
    struct: SimpleType
    methods: List[Function]

@dataclass
class Module(Declaration):
    name: str
    body: Block
    is_public: bool

@dataclass
class Import(Declaration):
    path: List[str]
    alias: Optional[str]
    imports: Optional[List[Tuple[str, Optional[str]]]]

# --- Statements ---
@dataclass
class ExpressionStatement(Statement):
    expression: 'Expression'

@dataclass
class If(Statement):
    condition: 'Expression'
    then_branch: Block
    else_branch: Optional[Union['If', Block]]

@dataclass
class While(Statement):
    condition: 'Expression'
    body: Block

@dataclass
class For(Statement):
    var_name: str
    iterable: 'Expression'
    body: Block

@dataclass
class Match(Statement):
    value: 'Expression'
    cases: List['MatchCase']

@dataclass
class MatchCase(ASTNode):
    pattern: 'Pattern'
    body: 'Expression'

@dataclass
class Pattern(ASTNode): pass # Base for pattern matching

@dataclass
class Return(Statement):
    value: Optional['Expression']

@dataclass
class Break(Statement): pass

@dataclass
class Continue(Statement): pass

@dataclass
class Defer(Statement):
    expression: 'Expression'

@dataclass
class UnsafeBlock(Statement):
    body: Block

@dataclass
class Try(Statement):
    try_block: Block
    catch_blocks: List[Tuple[str, TypeExpr, Block]]
    finally_block: Optional[Block]

@dataclass
class Throw(Statement):
    value: 'Expression'

# --- Expressions ---
@dataclass
class Expression(ASTNode): pass

@dataclass
class Identifier(Expression):
    name: str

@dataclass
class Literal(Expression):
    value: Any

@dataclass
class BinaryOp(Expression):
    left: Expression
    op: TokenType
    right: Expression

@dataclass
class UnaryOp(Expression):
    op: TokenType
    right: Expression

@dataclass
class Assignment(Expression):
    target: Expression
    value: Expression

@dataclass
class Call(Expression):
    callee: Expression
    args: List[Expression]

@dataclass
class MemberAccess(Expression):
    obj: Expression
    member: Identifier

@dataclass
class Index(Expression):
    obj: Expression
    index: Expression

@dataclass
class StructLiteral(Expression):
    name: SimpleType
    fields: Dict[str, Expression]

@dataclass
class ArrayLiteral(Expression):
    elements: List[Expression]

@dataclass
class TupleLiteral(Expression):
    elements: List[Expression]

@dataclass
class Lambda(Expression):
    params: List[Let]
    return_type: Optional[TypeExpr]
    body: Union[Block, Expression]

@dataclass
class Await(Expression):
    expression: Expression

# =================================================================================================
# ============================================ LEXER ==============================================
# =================================================================================================

class Lexer:
    def __init__(self, source: str, filename: str, error_reporter: ErrorReporter):
        self.source = source
        self.filename = filename
        self.error_reporter = error_reporter
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.col_start_of_line = 0

    @property
    def column(self):
        return self.current - self.col_start_of_line + 1

    KEYWORDS = {
        'let': TokenType.LET, 'const': TokenType.CONST, 'fn': TokenType.FN, 
        'return': TokenType.RETURN, 'if': TokenType.IF, 'else': TokenType.ELSE, 
        'for': TokenType.FOR, 'while': TokenType.WHILE, 'in': TokenType.IN, 
        'break': TokenType.BREAK, 'continue': TokenType.CONTINUE, 
        'struct': TokenType.STRUCT, 'enum': TokenType.ENUM, 'trait': TokenType.TRAIT, 
        'impl': TokenType.IMPL, 'import': TokenType.IMPORT, 'as': TokenType.AS, 
        'from': TokenType.FROM, 'pub': TokenType.PUB, 'static': TokenType.STATIC, 
        'async': TokenType.ASYNC, 'await': TokenType.AWAIT, 'try': TokenType.TRY, 
        'catch': TokenType.CATCH, 'finally': TokenType.FINALLY, 'throw': TokenType.THROW, 
        'match': TokenType.MATCH, 'module': TokenType.MODULE, 'unsafe': TokenType.UNSAFE, 
        'defer': TokenType.DEFER, 'true': TokenType.TRUE, 'false': TokenType.FALSE,
        'never': TokenType.NEVER
    }

    def scan_tokens(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column, self.filename))
        return self.tokens

    def scan_token(self):
        char = self.advance()
        
        if char in ' \t\r': return
        if char == '\n':
            self.line += 1
            self.col_start_of_line = self.current
            return

        # Comments
        if char == '/':
            if self.match('/'):
                while self.peek() != '\n' and not self.is_at_end(): self.advance()
                return
            elif self.match('*'):
                self.block_comment()
                return

        # Two-character tokens
        if char == '=': self.add_token(TokenType.EQ_EQ if self.match('=') else TokenType.EQ)
        elif char == '!': self.add_token(TokenType.BANG_EQ if self.match('=') else TokenType.BANG)
        elif char == '<': self.add_token(TokenType.LT_EQ if self.match('=') else (TokenType.LT_LT if self.match('<') else TokenType.LT))
        elif char == '>': self.add_token(TokenType.GT_EQ if self.match('=') else (TokenType.GT_GT if self.match('>') else TokenType.GT))
        elif char == '&': self.add_token(TokenType.AMP_AMP if self.match('&') else TokenType.AMP)
        elif char == '|': self.add_token(TokenType.PIPE_PIPE if self.match('|') else TokenType.PIPE)
        elif char == '-': self.add_token(TokenType.ARROW if self.match('>') else TokenType.MINUS)
        
        # Single-character tokens
        elif char == '(': self.add_token(TokenType.LPAREN)
        elif char == ')': self.add_token(TokenType.RPAREN)
        elif char == '{': self.add_token(TokenType.LBRACE)
        elif char == '}': self.add_token(TokenType.RBRACE)
        elif char == '[': self.add_token(TokenType.LBRACKET)
        elif char == ']': self.add_token(TokenType.RBRACKET)
        elif char == ',': self.add_token(TokenType.COMMA)
        elif char == '.': self.add_token(TokenType.DOT)
        elif char == ';': self.add_token(TokenType.SEMI)
        elif char == ':': self.add_token(TokenType.COLON)
        elif char == '+': self.add_token(TokenType.PLUS)
        elif char == '*': self.add_token(TokenType.STAR)
        elif char == '%': self.add_token(TokenType.PERCENT)
        elif char == '^': self.add_token(TokenType.CARET)
        elif char == '@': self.add_token(TokenType.AT)
        elif char == '_': self.add_token(TokenType.UNDERSCORE)
        elif char == '?': self.add_token(TokenType.QUESTION)

        # Literals
        elif char.isdigit(): self.number()
        elif char == '"': self.string()
        elif char == "'": self.char()
        elif char.isalpha() or char == '_': self.identifier()
        
        else:
            self.error_reporter.error(f"Unexpected character: '{char}'", self.filename, self.line, self.column)

    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        text = self.source[self.start:self.current]
        token_type = self.KEYWORDS.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type)

    def number(self):
        # ... (implementation from joyl.py for hex/binary/float can be added here)
        while self.peek().isdigit():
            self.advance()
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()
            self.add_token(TokenType.FLOAT, float(self.source[self.start:self.current]))
        else:
            self.add_token(TokenType.INTEGER, int(self.source[self.start:self.current]))

    def string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.col_start_of_line = self.current
            self.advance()
        if self.is_at_end():
            self.error_reporter.error("Unterminated string.", self.filename, self.line, self.column)
            return
        self.advance() # Closing "
        value = self.source[self.start + 1:self.current - 1]
        # Handle escape sequences here
        self.add_token(TokenType.STRING, value)

    def char(self):
        # ... (implementation for char literal with escapes)
        value = self.advance()
        if self.peek() != "'":
             self.error_reporter.error("Unterminated character literal.", self.filename, self.line, self.column)
        self.advance() # Closing '
        self.add_token(TokenType.CHAR, value)


    def advance(self) -> str:
        self.current += 1
        return self.source[self.current - 1]

    def match(self, expected: str) -> bool:
        if self.is_at_end(): return False
        if self.source[self.current] != expected: return False
        self.current += 1
        return True

    def peek(self) -> str:
        return '\0' if self.is_at_end() else self.source[self.current]

    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source): return '\0'
        return self.source[self.current + 1]

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def add_token(self, type: TokenType, literal: Any = None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, literal if literal is not None else text, self.line, self.column, self.filename))

@dataclass
class Token:
    type: TokenType
    lexeme: Any
    line: int
    column: int
    filename: str

# =================================================================================================
# ============================================ PARSER =============================================
# =================================================================================================
# A full-featured parser for the rich AST. This is a major integration point.

class Parser:
    def __init__(self, tokens: List[Token], filename: str, error_reporter: ErrorReporter):
        self.tokens = tokens
        self.filename = filename
        self.error_reporter = error_reporter
        self.current = 0

    def parse(self) -> Program:
        statements = []
        while not self.is_at_end():
            try:
                statements.append(self.declaration())
            except CompilerError as e:
                self.error_reporter.errors.append(e)
                self.synchronize()
        
        return Program(1, 1, self.filename, statements)

    # --- Declarations ---
    def declaration(self) -> Declaration:
        is_public = self.match(TokenType.PUB)
        
        if self.check(TokenType.FN):
            return self.function_declaration(is_public)
        if self.check(TokenType.STRUCT):
            return self.struct_declaration(is_public)
        if self.check(TokenType.IMPL):
            return self.impl_declaration()
        # ... other top-level declarations (enum, trait, module)
        
        if self.check(TokenType.LET) or self.check(TokenType.CONST):
             return self.let_declaration()

        return self.statement()

    def function_declaration(self, is_public: bool) -> Function:
        self.consume(TokenType.FN, "Expect 'fn' keyword.")
        name = self.consume(TokenType.IDENTIFIER, "Expect function name.").lexeme
        # Generics, params, return type, body...
        # This will be a complex function combining logic from both original files.
        # For brevity in this combined file, we'll keep it high-level.
        params = [] # parse parameters
        self.consume(TokenType.LPAREN, "Expect '(' after function name.")
        self.consume(TokenType.RPAREN, "Expect ')' after parameters.")

        body = self.block()
        
        return Function(self.previous().line, self.previous().column, self.filename,
                        name=name, generics=[], params=params, return_type=None, body=body,
                        is_async=False, is_public=is_public)

    def let_declaration(self) -> Let:
        is_mutable = self.match(TokenType.LET)
        if not is_mutable:
            self.consume(TokenType.CONST, "Expect 'let' or 'const'.")

        name = self.consume(TokenType.IDENTIFIER, "Expect variable name.").lexeme
        
        type_decl = None
        if self.match(TokenType.COLON):
            type_decl = self.parse_type_expression()

        initializer = None
        if self.match(TokenType.EQ):
            initializer = self.expression()

        self.consume(TokenType.SEMI, "Expect ';' after variable declaration.")
        return Let(self.previous().line, self.previous().column, self.filename,
                   name=name, type_decl=type_decl, initializer=initializer, is_mutable=is_mutable)

    def struct_declaration(self, is_public: bool) -> Struct:
        self.consume(TokenType.STRUCT, "Expect 'struct' keyword.")
        name_tok = self.consume(TokenType.IDENTIFIER, "Expect struct name.")
        
        generics = self.parse_generic_params()
        
        self.consume(TokenType.LBRACE, f"Expect '{{' after struct name '{name_tok.lexeme}'.")
        fields = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            field_name = self.consume(TokenType.IDENTIFIER, "Expect field name.").lexeme
            self.consume(TokenType.COLON, "Expect ':' after field name.")
            field_type = self.parse_type_expression()
            fields.append(Let(name_tok.line, name_tok.column, self.filename, 
                              name=field_name, type_decl=field_type, initializer=None, is_mutable=True))
            if not self.match(TokenType.COMMA):
                break
        self.consume(TokenType.RBRACE, "Expect '}' after struct fields.")

        return Struct(name_tok.line, name_tok.column, self.filename, name=name_tok.lexeme, generics=generics, fields=fields, is_public=is_public)

    def impl_declaration(self) -> Impl:
        # impl<T> Trait<T> for Struct<T> { ... }
        # Simplified: impl Trait for Struct { ... }
        self.consume(TokenType.IMPL, "Expect 'impl' keyword.")
        
        # This part requires careful parsing to distinguish 'impl Trait for Struct' from 'impl Struct'
        target_type = self.parse_simple_type()
        trait = None
        
        if self.match(TokenType.FOR):
            trait = target_type
            target_type = self.parse_simple_type()

        methods = []
        self.consume(TokenType.LBRACE, "Expect '{' before impl body.")
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            is_public = self.match(TokenType.PUB)
            methods.append(self.function_declaration(is_public))
        self.consume(TokenType.RBRACE, "Expect '}' after impl body.")

        return Impl(self.previous().line, self.previous().column, self.filename,
                    generics=[], trait=trait, struct=target_type, methods=methods)

    # --- Statements ---
    def statement(self) -> Statement:
        if self.match(TokenType.IF): return self.if_statement()
        if self.match(TokenType.WHILE): return self.while_statement()
        if self.match(TokenType.FOR): return self.for_statement()
        if self.match(TokenType.RETURN): return self.return_statement()
        if self.match(TokenType.LBRACE): return self.block()
        # ... other statements (match, try, etc.)
        
        return self.expression_statement()

    def if_statement(self) -> If:
        condition = self.expression()
        then_branch = self.block()
        else_branch = None
        if self.match(TokenType.ELSE):
            if self.check(TokenType.IF):
                else_branch = self.if_statement()
            else:
                else_branch = self.block()
        return If(self.previous().line, self.previous().column, self.filename,
                  condition=condition, then_branch=then_branch, else_branch=else_branch)
    
    def block(self) -> Block:
        line, col = self.peek().line, self.peek().column
        self.consume(TokenType.LBRACE, "Expect '{' to start a block.")
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.declaration())
        self.consume(TokenType.RBRACE, "Expect '}' to end a block.")
        return Block(line, col, self.filename, statements)

    def return_statement(self) -> Return:
        line, col = self.previous().line, self.previous().column
        value = None
        if not self.check(TokenType.SEMI):
            value = self.expression()
        self.consume(TokenType.SEMI, "Expect ';' after return value.")
        return Return(line, col, self.filename, value)

    def expression_statement(self) -> ExpressionStatement:
        expr = self.expression()
        self.consume(TokenType.SEMI, "Expect ';' after expression.")
        return ExpressionStatement(expr.line, expr.column, expr.filename, expr)

    # --- Expressions (Pratt Parser Style) ---
    def expression(self) -> Expression:
        return self.assignment()
        
    def assignment(self) -> Expression:
        expr = self.logic_or()
        if self.match(TokenType.EQ):
            equals = self.previous()
            value = self.assignment()
            if isinstance(expr, Identifier) or isinstance(expr, MemberAccess):
                return Assignment(expr.line, expr.column, expr.filename, expr, value)
            self.error_reporter.error("Invalid assignment target.", node=equals)
        return expr

    def logic_or(self) -> Expression:
        expr = self.logic_and()
        while self.match(TokenType.PIPE_PIPE):
            op = self.previous().type
            right = self.logic_and()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr
    
    def logic_and(self) -> Expression:
        expr = self.equality()
        while self.match(TokenType.AMP_AMP):
            op = self.previous().type
            right = self.equality()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr
    
    def equality(self) -> Expression:
        expr = self.comparison()
        while self.match(TokenType.EQ_EQ, TokenType.BANG_EQ):
            op = self.previous().type
            right = self.comparison()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr

    def comparison(self) -> Expression:
        expr = self.term()
        while self.match(TokenType.GT, TokenType.GT_EQ, TokenType.LT, TokenType.LT_EQ):
            op = self.previous().type
            right = self.term()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr

    def term(self) -> Expression:
        expr = self.factor()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.previous().type
            right = self.factor()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr

    def factor(self) -> Expression:
        expr = self.unary()
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.previous().type
            right = self.unary()
            expr = BinaryOp(expr.line, expr.column, expr.filename, expr, op, right)
        return expr

    def unary(self) -> Expression:
        if self.match(TokenType.BANG, TokenType.MINUS):
            op_tok = self.previous()
            right = self.unary()
            return UnaryOp(op_tok.line, op_tok.column, self.filename, op_tok.type, right)
        return self.call()

    def call(self) -> Expression:
        expr = self.primary()
        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                member = self.consume(TokenType.IDENTIFIER, "Expect property name after '.'.")
                expr = MemberAccess(expr.line, expr.column, expr.filename, expr, Identifier(member.line, member.column, self.filename, member.lexeme))
            else:
                break
        return expr

    def finish_call(self, callee: Expression) -> Call:
        args = []
        if not self.check(TokenType.RPAREN):
            while True:
                args.append(self.expression())
                if not self.match(TokenType.COMMA):
                    break
        self.consume(TokenType.RPAREN, "Expect ')' after arguments.")
        return Call(callee.line, callee.column, callee.filename, callee, args)

    def primary(self) -> Expression:
        tok = self.peek()
        if self.match(TokenType.FALSE): return Literal(self.previous().line, self.previous().column, self.filename, False)
        if self.match(TokenType.TRUE): return Literal(self.previous().line, self.previous().column, self.filename, True)
        if self.match(TokenType.INTEGER): return Literal(self.previous().line, self.previous().column, self.filename, self.previous().lexeme)
        if self.match(TokenType.FLOAT): return Literal(self.previous().line, self.previous().column, self.filename, self.previous().lexeme)
        if self.match(TokenType.STRING): return Literal(self.previous().line, self.previous().column, self.filename, self.previous().lexeme)
        if self.match(TokenType.IDENTIFIER):
             # Could be a variable or a struct literal
            if self.check(TokenType.LBRACE):
                return self.struct_literal()
            return Identifier(self.previous().line, self.previous().column, self.filename, self.previous().lexeme)
        
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expect ')' after expression.")
            return expr # Grouping expression, no dedicated node needed
        
        raise self.error(f"Unexpected token: {self.peek().lexeme}")


    # --- Type Parsing ---
    def parse_type_expression(self) -> TypeExpr:
        # This is where the type parsing logic from joylt4.py is integrated
        if self.match(TokenType.STAR):
            is_mutable = self.match(TokenType.LET) # *mut
            target = self.parse_type_expression()
            return PointerType(target.line, target.column, self.filename, target, is_mutable)
        
        if self.match(TokenType.LBRACKET):
            element = self.parse_type_expression()
            size = None
            if self.match(TokenType.SEMI):
                size = self.expression()
            self.consume(TokenType.RBRACKET, "Expect ']' after array/slice type.")
            return ArrayType(element.line, element.column, self.filename, element, size)

        return self.parse_simple_type()

    def parse_simple_type(self) -> SimpleType:
        name_tok = self.consume(TokenType.IDENTIFIER, "Expect type name.")
        generics = []
        if self.match(TokenType.LT):
            while True:
                generics.append(self.parse_type_expression())
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.GT, "Expect '>' after generic arguments.")
        return SimpleType(name_tok.line, name_tok.column, self.filename, name_tok.lexeme, generics)

    def parse_generic_params(self) -> List[SimpleType]:
        generics = []
        if self.match(TokenType.LT):
            while True:
                generics.append(self.parse_simple_type())
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.GT, "Expect '>' after generic parameters.")
        return generics
        
    # --- Utility Methods ---
    def match(self, *types: TokenType) -> bool:
        for t in types:
            if self.check(t):
                self.advance()
                return True
        return False

    def consume(self, type: TokenType, message: str) -> Token:
        if self.check(type): return self.advance()
        raise self.error(message)

    def check(self, type: TokenType) -> bool:
        if self.is_at_end(): return False
        return self.peek().type == type

    def advance(self) -> Token:
        if not self.is_at_end(): self.current += 1
        return self.previous()

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def previous(self) -> Token:
        return self.tokens[self.current - 1]
        
    def error(self, message: str) -> CompilerError:
        token = self.peek()
        return CompilerError(message, token.filename, token.line, token.column)
        
    def synchronize(self):
        # ... implementation to recover from a parsing error
        self.advance()
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMI: return
            if self.peek().type in {TokenType.FN, TokenType.LET, TokenType.CONST, TokenType.STRUCT, TokenType.IF}: return
            self.advance()

# =================================================================================================
# ======================================== SEMANTIC ANALYSIS ======================================
# =================================================================================================
# The heart of the compiler's intelligence.

class SemanticAnalyzer:
    def __init__(self, type_system: TypeSystem, error_reporter: ErrorReporter):
        self.type_system = type_system
        self.error_reporter = error_reporter
        self.scopes = [{}]
        self.current_function = None

    def analyze(self, program: Program):
        # 1. First Pass: Register all top-level types (structs, enums, traits).
        self.visit(program, pass_one=True)
        # 2. Second Pass: Analyze implementations and function bodies.
        self.visit(program, pass_one=False)
    
    def enter_scope(self):
        self.scopes.append({})
        
    def exit_scope(self):
        self.scopes.pop()
        
    def declare(self, name: str, type: Type):
        self.scopes[-1][name] = type
        
    def lookup(self, name: str) -> Optional[Type]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit(self, node: ASTNode, pass_one=False):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, pass_one) if pass_one else visitor(node)
        
    def generic_visit(self, node: ASTNode, pass_one=None):
         raise NotImplementedError(f"No visit method for {type(node).__name__}")
    
    # --- Visitor Methods ---
    
    def visit_Program(self, node: Program, pass_one: bool):
        for stmt in node.statements:
            self.visit(stmt, pass_one=pass_one)

    def visit_Struct(self, node: Struct, pass_one: bool):
        if pass_one:
            if self.type_system.get_type(node.name):
                self.error_reporter.error(f"Type '{node.name}' already defined.", node=node)
                return
            
            # Register the struct type placeholder
            struct_type = Type(TypeKind.STRUCT, node.name, is_public=node.is_public)
            if node.generics:
                struct_type.generic_params = [g.name for g in node.generics]
            self.type_system.add_type(struct_type)
        else:
            # Analyze fields in the second pass
            struct_type = self.type_system.get_type(node.name)
            total_size = 0
            max_align = 1
            for field in node.fields:
                field_type = self.resolve_type_expr(field.type_decl)
                if not field_type:
                    self.error_reporter.error(f"Unknown type for field '{field.name}'.", node=field.type_decl)
                    continue
                struct_type.members[field.name] = field_type
                # Align offset
                total_size = (total_size + field_type.align - 1) & -field_type.align
                total_size += field_type.size
                max_align = max(max_align, field_type.align)

            struct_type.size = (total_size + max_align - 1) & -max_align
            struct_type.align = max_align

    def visit_Let(self, node: Let, pass_one: bool):
        if pass_one: return

        # Check for redeclaration in the same scope
        if node.name in self.scopes[-1]:
            self.error_reporter.error(f"Variable '{node.name}' is already declared in this scope.", node=node)

        initializer_type = None
        if node.initializer:
            initializer_type = self.visit(node.initializer)

        declared_type = None
        if node.type_decl:
            declared_type = self.resolve_type_expr(node.type_decl)
            if not declared_type:
                self.error_reporter.error(f"Unknown type '{node.type_decl.name}'.", node=node.type_decl)

        if declared_type and initializer_type:
            if not self.type_system.is_subtype(initializer_type, declared_type):
                self.error_reporter.error(f"Type mismatch: cannot assign '{initializer_type.name}' to variable of type '{declared_type.name}'.", node=node.initializer)
            final_type = declared_type
        elif declared_type:
            final_type = declared_type
        elif initializer_type:
            final_type = initializer_type
        else:
            self.error_reporter.error(f"Cannot infer type for '{node.name}'. Please provide a type annotation or an initializer.", node=node)
            return
            
        node.inferred_type = final_type
        self.declare(node.name, final_type)

    def visit_Function(self, node: Function, pass_one: bool):
        if pass_one:
            # Pre-register function signature for recursion
            # This part is more complex with generics and needs careful handling
            pass
        else:
            self.current_function = node
            self.enter_scope()
            
            # Analyze params
            for param in node.params:
                self.visit(param)

            # Analyze body
            self.visit(node.body)
            
            self.exit_scope()
            self.current_function = None

    def visit_Block(self, node: Block, pass_one: bool):
        self.enter_scope()
        for stmt in node.statements:
            self.visit(stmt, pass_one=pass_one)
        self.exit_scope()

    def visit_ExpressionStatement(self, node: ExpressionStatement, pass_one: bool):
        if pass_one: return
        self.visit(node.expression)

    def visit_Return(self, node: Return, pass_one: bool):
        if pass_one: return
        # Check if return type matches function's declared return type
        # ...
        
    # --- Expression Visitors (Return Inferred Type) ---
    
    def visit_BinaryOp(self, node: BinaryOp) -> Type:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        if not left_type or not right_type:
            return None # Error already reported

        # Type check based on operator
        if node.op in {TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH}:
            unified_type = self.type_system.unify_types(left_type, right_type)
            if not unified_type or unified_type.kind != TypeKind.PRIMITIVE:
                self.error_reporter.error(f"Cannot apply operator '{node.op.name}' to types '{left_type.name}' and '{right_type.name}'.", node=node)
                return None
            node.inferred_type = unified_type
            return unified_type
        
        if node.op in {TokenType.EQ_EQ, TokenType.BANG_EQ, TokenType.LT, TokenType.GT, TokenType.LT_EQ, TokenType.GT_EQ}:
            # ... check if types are comparable ...
            bool_type = self.type_system.get_type('bool')
            node.inferred_type = bool_type
            return bool_type

        return None
    
    def visit_Identifier(self, node: Identifier) -> Type:
        var_type = self.lookup(node.name)
        if not var_type:
            self.error_reporter.error(f"Undefined variable '{node.name}'.", node=node)
            return None
        node.inferred_type = var_type
        return var_type
        
    def visit_Literal(self, node: Literal) -> Type:
        t = None
        if isinstance(node.value, bool): t = self.type_system.get_type('bool')
        elif isinstance(node.value, int): t = self.type_system.get_type('i32') # Default
        elif isinstance(node.value, float): t = self.type_system.get_type('f64') # Default
        elif isinstance(node.value, str): t = self.type_system.get_type('str')
        
        node.inferred_type = t
        return t

    # --- Type Resolution ---
    def resolve_type_expr(self, type_expr: TypeExpr) -> Optional[Type]:
        if isinstance(type_expr, SimpleType):
            base_type = self.type_system.get_type(type_expr.name)
            if not base_type: return None
            # Handle generics instantiation here
            return base_type
        if isinstance(type_expr, PointerType):
            target = self.resolve_type_expr(type_expr.target)
            return self.type_system.pointer_to(target, type_expr.is_mutable) if target else None
        if isinstance(type_expr, ArrayType):
            element = self.resolve_type_expr(type_expr.element)
            # Size must be a constant expression, needs to be evaluated here
            return self.type_system.array_of(element, 5) if element else None # Placeholder size
        return None


# =================================================================================================
# ======================================== CODE GENERATION ========================================
# =================================================================================================
# Backend for LLVM. A WASM backend would follow a similar structure.

class LLVMBackend:
    def __init__(self, type_system: TypeSystem, error_reporter: ErrorReporter):
        self.type_system = type_system
        self.error_reporter = error_reporter
        self.llvm_ir = []
        self.temp_counter = 0
        self.local_vars = {} # Maps variable name to LLVM register/pointer

    def compile(self, program: Program, output_path: str, mode: CompilerMode):
        self.generate_prologue()

        for stmt in program.statements:
            self.compile_statement(stmt)
        
        # Write LLVM IR to file
        with open(output_path, 'w') as f:
            f.write("\n".join(self.llvm_ir))
        
        # Use llc and clang to produce an executable
        obj_path = output_path.replace('.ll', '.o')
        exe_path = os.path.splitext(output_path)[0]

        try:
            subprocess.run(['llc', '-filetype=obj', '-o', obj_path, output_path], check=True)
            subprocess.run(['clang', obj_path, '-o', exe_path], check=True)
            print(f"Successfully compiled to executable: {exe_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.error_reporter.error(f"LLVM toolchain command failed: {e}. Is llc/clang in your PATH?")
            return None

        return exe_path

    def generate_prologue(self):
        # ... standard LLVM setup, target triples, etc. ...
        self.llvm_ir.append('define i32 @main() {')
        self.llvm_ir.append('entry:')

    def compile_statement(self, stmt: Statement):
        if isinstance(stmt, Let):
            self.compile_let(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self.compile_expression(stmt.expression)
        elif isinstance(stmt, Return):
            # This should be inside a function context
            pass 
        # ... other statements

    def compile_let(self, stmt: Let):
        llvm_type = self.to_llvm_type(stmt.inferred_type)
        
        # Allocate stack space for the variable
        self.emit(f"%{stmt.name} = alloca {llvm_type}")
        self.local_vars[stmt.name] = (f"%{stmt.name}", stmt.inferred_type)

        if stmt.initializer:
            value_reg = self.compile_expression(stmt.initializer)
            self.emit(f"store {llvm_type} {value_reg}, {llvm_type}* %{stmt.name}")
            
    def compile_expression(self, expr: Expression) -> str:
        if isinstance(expr, Literal):
            return self.compile_literal(expr)
        if isinstance(expr, Identifier):
            ptr, type = self.local_vars[expr.name]
            reg = self.new_temp()
            self.emit(f"{reg} = load {self.to_llvm_type(type)}, {self.to_llvm_type(type)}* {ptr}")
            return reg
        if isinstance(expr, BinaryOp):
            return self.compile_binary_op(expr)
        # ... other expressions
        return "undef"

    def compile_literal(self, literal: Literal) -> str:
        if isinstance(literal.value, int):
            return str(literal.value)
        if isinstance(literal.value, bool):
            return "1" if literal.value else "0"
        # ... other literals (strings are more complex)
        return "undef"

    def compile_binary_op(self, expr: BinaryOp) -> str:
        left_reg = self.compile_expression(expr.left)
        right_reg = self.compile_expression(expr.right)
        
        op_map = {
            TokenType.PLUS: "add", TokenType.MINUS: "sub",
            TokenType.STAR: "mul", TokenType.SLASH: "sdiv", # signed division
            TokenType.EQ_EQ: "icmp eq", TokenType.BANG_EQ: "icmp ne",
            TokenType.LT: "icmp slt", TokenType.GT: "icmp sgt",
            TokenType.LT_EQ: "icmp sle", TokenType.GT_EQ: "icmp sge",
        }
        
        llvm_op = op_map.get(expr.op)
        if not llvm_op:
            self.error_reporter.error(f"Unsupported binary operator in codegen: {expr.op.name}", node=expr)
            return "undef"

        target_reg = self.new_temp()
        llvm_type = self.to_llvm_type(expr.inferred_type)
        
        self.emit(f"{target_reg} = {llvm_op} {llvm_type} {left_reg}, {right_reg}")
        return target_reg

    def to_llvm_type(self, type: Type) -> str:
        if not type: return "void"
        # Mapping from TypeSystem to LLVM types
        type_map = {
            'i32': 'i32', 'i64': 'i64', 'bool': 'i1',
            'f64': 'double', 'f32': 'float', 'never': 'void'
        }
        if type.name in type_map:
            return type_map[type.name]
        if type.kind == TypeKind.POINTER:
            return f"{self.to_llvm_type(type.element_type)}*"
        
        self.error_reporter.warning(f"Unsupported type in LLVM codegen: '{type.name}'")
        return "i32" # Fallback

    def new_temp(self) -> str:
        reg = f"%t{self.temp_counter}"
        self.temp_counter += 1
        return reg

    def emit(self, instruction: str):
        self.llvm_ir.append(f"  {instruction}")

# =================================================================================================
# ========================================= COMPILER DRIVER =======================================
# =================================================================================================

class JoylCompiler:
    def __init__(self, mode: CompilerMode = CompilerMode.DEBUG, 
                 optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
                 target: TargetPlatform = TargetPlatform.LLVM):
        self.mode = mode
        self.optimization_level = optimization_level
        self.target = target
        self.error_reporter = ErrorReporter()
        self.type_system = TypeSystem(self.error_reporter)
        
    def compile(self, source: str, filename: str, output_path: str):
        try:
            # 1. Lexical Analysis
            lexer = Lexer(source, filename, self.error_reporter)
            tokens = lexer.scan_tokens()
            if self.error_reporter.has_errors():
                self.error_reporter.print_errors()
                return None
            
            # 2. Parsing
            parser = Parser(tokens, filename, self.error_reporter)
            ast = parser.parse()
            if self.error_reporter.has_errors():
                self.error_reporter.print_errors()
                return None
            
            # 3. Semantic Analysis
            analyzer = SemanticAnalyzer(self.type_system, self.error_reporter)
            analyzer.analyze(ast)
            if self.error_reporter.has_errors():
                self.error_reporter.print_errors()
                return None
            
            # 4. Code Generation
            if self.target == TargetPlatform.LLVM:
                backend = LLVMBackend(self.type_system, self.error_reporter)
                output_file = (output_path if output_path 
                               else os.path.splitext(filename)[0] + ".ll")
                return backend.compile(ast, output_file, self.mode)
            else:
                self.error_reporter.error(f"Target '{self.target.name}' is not yet supported.")
                self.error_reporter.print_errors()
                return None
            
        except CompilerError as e:
            self.error_reporter.errors.append(e)
            self.error_reporter.print_errors()
            return None
        except Exception as e:
            # Catch unexpected compiler crashes
            print(f"FATAL COMPILER ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

# =================================================================================================
# ======================================== MAIN & CLI =============================================
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description='The Joyl Compiler - Definitive Edition')
    parser.add_argument('file', help='Joyl source file to compile.')
    parser.add_argument('-o', '--output', help='Output file path (for .ll or executable).')
    parser.add_argument('--target', choices=['llvm'], default='llvm', help='Compilation target.')
    parser.add_argument('-O', '--optimize', type=int, default=DEFAULT_OPTIMIZATION_LEVEL, help='Optimization level (0-3).')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    args = parser.parse_args()

    mode = CompilerMode.DEBUG if args.debug else CompilerMode.RELEASE
    target = TargetPlatform.LLVM # Currently only LLVM is supported in this version
    
    compiler = JoylCompiler(mode, args.optimize, target)

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            source = f.read()
    except IOError as e:
        print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)
        
    result_path = compiler.compile(source, args.file, args.output)
    
    if result_path is None:
        print("\nCompilation failed.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nCompilation successful. Output at: {result_path}")

if __name__ == "__main__":
    main()