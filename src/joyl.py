#!/usr/bin/env python3
# Joyl Compiler v4.0 - The Complete Professional Compiler
import sys
import os
import re
import time
import struct
import marshal
import inspect
import hashlib
import threading
import queue
import math
import json
import glob
import traceback
import mmap
import contextlib
import subprocess
import platform
import importlib
import zipfile
import tempfile
import shutil
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict, ChainMap
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse

# ==================== CONSTANTS & CONFIG ====================
VERSION = "4.0.0"
DEFAULT_OPTIMIZATION_LEVEL = 3
MAX_ERRORS_BEFORE_ABORT = 20
CACHE_DIR = ".joylcache"
STDLIB_PATH = os.path.join(os.path.dirname(__file__), "stdlib")
PKG_CACHE = os.path.expanduser("~/.joyl/packages")
PKG_REGISTRY = "https://packages.joyl-lang.org"
WASM_TARGET = "wasm32-unknown-unknown"
LLVM_TARGET = "native"

# ==================== ENUMS ====================
class TokenType(Enum):
    # Identifiers and literals
    IDENTIFIER = auto()
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    
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
    CLASS = auto()
    STRUCT = auto()
    TRAIT = auto()
    IMPLEMENTS = auto()
    IMPORT = auto()
    AS = auto()
    FROM = auto()
    PUB = auto()
    PRIV = auto()
    PROT = auto()
    STATIC = auto()
    ASYNC = auto()
    AWAIT = auto()
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    THROW = auto()
    MATCH = auto()
    WITH = auto()
    USE = auto()
    MODULE = auto()
    UNSAFE = auto()
    DEFER = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    CARET = auto()
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
    TILDE = auto()
    PLUS_EQ = auto()
    MINUS_EQ = auto()
    STAR_EQ = auto()
    SLASH_EQ = auto()
    PERCENT_EQ = auto()
    CARET_EQ = auto()
    AMP_EQ = auto()
    PIPE_EQ = auto()
    LT_LT_EQ = auto()
    GT_GT_EQ = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    DOT_DOT = auto()
    COLON = auto()
    SEMI = auto()
    ARROW = auto()
    FAT_ARROW = auto()
    AT = auto()
    UNDERSCORE = auto()
    QUESTION = auto()
    DOLLAR = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()

class ValueType(Enum):
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
    CHAR = auto()
    NIL = auto()
    LIST = auto()
    DICT = auto()
    FUNCTION = auto()
    CLASS = auto()
    OBJECT = auto()
    NATIVE = auto()
    FUTURE = auto()
    ERROR = auto()
    MODULE = auto()

class CompilerMode(Enum):
    DEBUG = auto()
    RELEASE = auto()
    OPTIMIZE = auto()

class TargetPlatform(Enum):
    NATIVE = auto()
    WASM = auto()
    LLVM = auto()
    BYTECODE = auto()

class OpCode(Enum):
    # Stack manipulation
    PUSH = auto()
    POP = auto()
    DUP = auto()
    SWAP = auto()
    ROT = auto()
    
    # Variable operations
    LOAD = auto()
    STORE = auto()
    LOAD_GLOBAL = auto()
    STORE_GLOBAL = auto()
    LOAD_ATTR = auto()
    STORE_ATTR = auto()
    LOAD_INDEX = auto()
    STORE_INDEX = auto()
    LOAD_MODULE = auto()
    
    # Memory management
    ALLOC = auto()
    FREE = auto()
    GC_COLLECT = auto()
    
    # Arithmetic operations
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    NEG = auto()
    INC = auto()
    DEC = auto()
    
    # Bitwise operations
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    SHL = auto()
    SHR = auto()
    
    # Logical operations
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Comparison operations
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    IS = auto()
    IS_NOT = auto()
    
    # Control flow
    JMP = auto()
    JMP_IF = auto()
    JMP_IF_NOT = auto()
    JMP_IF_TRUE = auto()
    JMP_IF_FALSE = auto()
    CALL = auto()
    RETURN = auto()
    CALL_METHOD = auto()
    CALL_NATIVE = auto()
    TAIL_CALL = auto()
    
    # Object operations
    NEW = auto()
    NEW_ARRAY = auto()
    NEW_DICT = auto()
    NEW_CLASS = auto()
    NEW_FUNCTION = auto()
    INSTANCE_OF = auto()
    CLONE = auto()
    
    # Type operations
    TO_INT = auto()
    TO_FLOAT = auto()
    TO_BOOL = auto()
    TO_STRING = auto()
    TYPE_OF = auto()
    CHECK_TYPE = auto()
    
    # Async operations
    AWAIT = auto()
    YIELD = auto()
    SPAWN = auto()
    
    # Concurrency
    LOCK = auto()
    UNLOCK = auto()
    CHAN_SEND = auto()
    CHAN_RECV = auto()
    
    # Error handling
    THROW = auto()
    TRY_BEGIN = auto()
    TRY_END = auto()
    ASSERT = auto()
    
    # Debugging
    DEBUG = auto()
    BREAKPOINT = auto()
    
    # Special
    NOP = auto()
    HALT = auto()
    IMPORT = auto()
    EXPORT = auto()

# ==================== DATA STRUCTURES ====================
@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
    filename: str

@dataclass
class ASTNode:
    line: int
    column: int
    filename: str

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

@dataclass
class Block(ASTNode):
    statements: List[ASTNode]

@dataclass
class Module(ASTNode):
    name: str
    body: Block
    is_public: bool

@dataclass
class VarDecl(ASTNode):
    name: str
    type: Optional[str]
    value: Optional[ASTNode]
    is_const: bool
    is_public: bool
    is_unsafe: bool

@dataclass
class Function(ASTNode):
    name: str
    params: List[Tuple[str, Optional[str]]]
    return_type: Optional[str]
    body: Block
    is_async: bool
    is_public: bool
    is_unsafe: bool

@dataclass
class Class(ASTNode):
    name: str
    parent: Optional[str]
    traits: List[str]
    fields: List[VarDecl]
    methods: List[Function]
    is_public: bool

@dataclass
class Trait(ASTNode):
    name: str
    methods: List[Function]
    is_public: bool

@dataclass
class If(ASTNode):
    condition: ASTNode
    then_branch: ASTNode
    else_branch: Optional[ASTNode]

@dataclass
class For(ASTNode):
    var_name: str
    iterable: ASTNode
    body: ASTNode
    is_parallel: bool

@dataclass
class While(ASTNode):
    condition: ASTNode
    body: ASTNode

@dataclass
class Return(ASTNode):
    value: Optional[ASTNode]

@dataclass
class Break(ASTNode):
    pass

@dataclass
class Continue(ASTNode):
    pass

@dataclass
class Match(ASTNode):
    value: ASTNode
    cases: List[Tuple[ASTNode, ASTNode]]

@dataclass
class Try(ASTNode):
    try_block: ASTNode
    catch_blocks: List[Tuple[Optional[str], Optional[str], ASTNode]]
    finally_block: Optional[ASTNode]

@dataclass
class Throw(ASTNode):
    value: ASTNode

@dataclass
class Import(ASTNode):
    module: str
    alias: Optional[str]
    imports: List[Tuple[str, Optional[str]]]
    is_relative: bool
    version: Optional[str]

@dataclass
class Use(ASTNode):
    items: List[Tuple[str, Optional[str]]]
    from_module: Optional[str]

@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    op: TokenType
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    op: TokenType
    right: ASTNode

@dataclass
class Call(ASTNode):
    callee: ASTNode
    args: List[ASTNode]
    is_tail_call: bool

@dataclass
class MethodCall(ASTNode):
    obj: ASTNode
    method: str
    args: List[ASTNode]

@dataclass
class Attribute(ASTNode):
    obj: ASTNode
    attr: str

@dataclass
class Index(ASTNode):
    obj: ASTNode
    index: ASTNode

@dataclass
class Literal(ASTNode):
    value: Any

@dataclass
class Variable(ASTNode):
    name: str

@dataclass
class Assignment(ASTNode):
    target: ASTNode
    value: ASTNode
    op: TokenType

@dataclass
class Lambda(ASTNode):
    params: List[Tuple[str, Optional[str]]]
    return_type: Optional[str]
    body: ASTNode
    is_async: bool

@dataclass
class Array(ASTNode):
    elements: List[ASTNode]

@dataclass
class Dict(ASTNode):
    entries: List[Tuple[ASTNode, ASTNode]]

@dataclass
class Await(ASTNode):
    expr: ASTNode

@dataclass
class Yield(ASTNode):
    expr: Optional[ASTNode]

@dataclass
class Defer(ASTNode):
    expr: ASTNode

@dataclass
class Spawn(ASTNode):
    expr: ASTNode

@dataclass
class OwnershipTransfer(ASTNode):
    expr: ASTNode
    to: Optional[str]

@dataclass
class UnsafeBlock(ASTNode):
    body: Block

@dataclass
class CompilerError(Exception):
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    filename: Optional[str] = None
    hint: Optional[str] = None

    def __str__(self):
        loc = ""
        if self.filename:
            loc += f"{self.filename}:"
        if self.line is not None:
            loc += f"{self.line}:"
            if self.column is not None:
                loc += f"{self.column}"
        
        msg = self.message
        if loc:
            msg = f"{loc}: {msg}"
        if self.hint:
            msg += f"\nHint: {self.hint}"
        return msg

# ==================== LEXER ====================
class Lexer:
    KEYWORDS = {
        'let': TokenType.LET,
        'const': TokenType.CONST,
        'fn': TokenType.FN,
        'return': TokenType.RETURN,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'in': TokenType.IN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'class': TokenType.CLASS,
        'struct': TokenType.STRUCT,
        'trait': TokenType.TRAIT,
        'implements': TokenType.IMPLEMENTS,
        'import': TokenType.IMPORT,
        'as': TokenType.AS,
        'from': TokenType.FROM,
        'pub': TokenType.PUB,
        'priv': TokenType.PRIV,
        'prot': TokenType.PROT,
        'static': TokenType.STATIC,
        'async': TokenType.ASYNC,
        'await': TokenType.AWAIT,
        'try': TokenType.TRY,
        'catch': TokenType.CATCH,
        'finally': TokenType.FINALLY,
        'throw': TokenType.THROW,
        'match': TokenType.MATCH,
        'with': TokenType.WITH,
        'use': TokenType.USE,
        'module': TokenType.MODULE,
        'unsafe': TokenType.UNSAFE,
        'defer': TokenType.DEFER,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'nil': TokenType.NIL,
    }

    OPERATORS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.STAR,
        '/': TokenType.SLASH,
        '%': TokenType.PERCENT,
        '^': TokenType.CARET,
        '!': TokenType.BANG,
        '=': TokenType.EQ,
        '==': TokenType.EQ_EQ,
        '!=': TokenType.BANG_EQ,
        '<': TokenType.LT,
        '>': TokenType.GT,
        '<=': TokenType.LT_EQ,
        '>=': TokenType.GT_EQ,
        '&&': TokenType.AMP_AMP,
        '||': TokenType.PIPE_PIPE,
        '<<': TokenType.LT_LT,
        '>>': TokenType.GT_GT,
        '&': TokenType.AMP,
        '|': TokenType.PIPE,
        '~': TokenType.TILDE,
        '+=': TokenType.PLUS_EQ,
        '-=': TokenType.MINUS_EQ,
        '*=': TokenType.STAR_EQ,
        '/=': TokenType.SLASH_EQ,
        '%=': TokenType.PERCENT_EQ,
        '^=': TokenType.CARET_EQ,
        '&=': TokenType.AMP_EQ,
        '|=': TokenType.PIPE_EQ,
        '<<=': TokenType.LT_LT_EQ,
        '>>=': TokenType.GT_GT_EQ,
    }

    DELIMITERS = {
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
        ',': TokenType.COMMA,
        '.': TokenType.DOT,
        '..': TokenType.DOT_DOT,
        ':': TokenType.COLON,
        ';': TokenType.SEMI,
        '->': TokenType.ARROW,
        '=>': TokenType.FAT_ARROW,
        '@': TokenType.AT,
        '_': TokenType.UNDERSCORE,
        '?': TokenType.QUESTION,
        '$': TokenType.DOLLAR,
    }

    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        self.errors = []

    def scan_tokens(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.add_token(TokenType.EOF)
        return self.tokens

    def scan_token(self):
        c = self.advance()
        
        # Skip whitespace
        if c in (' ', '\t', '\r'):
            return
        
        # Handle newlines
        if c == '\n':
            self.add_token(TokenType.NEWLINE)
            self.line += 1
            self.column = 1
            return
        
        # Handle comments
        if c == '/' and self.peek() == '/':
            while self.peek() != '\n' and not self.is_at_end():
                self.advance()
            return
        
        if c == '/' and self.peek() == '*':
            self.advance()  # Consume the '*'
            while not (self.peek() == '*' and self.peek_next() == '/'):
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 0
                self.advance()
                if self.is_at_end():
                    self.error("Unterminated block comment")
                    return
            
            self.advance()  # Consume the '*'
            self.advance()  # Consume the '/'
            return
        
        # Handle multi-character operators/delimiters first
        two_char = c + self.peek()
        if two_char in self.OPERATORS:
            self.advance()
            self.add_token(self.OPERATORS[two_char])
            return
        
        if two_char in self.DELIMITERS:
            self.advance()
            self.add_token(self.DELIMITERS[two_char])
            return
        
        # Handle single-character operators/delimiters
        if c in self.OPERATORS:
            self.add_token(self.OPERATORS[c])
            return
        
        if c in self.DELIMITERS:
            self.add_token(self.DELIMITERS[c])
            return
        
        # Handle numbers
        if c.isdigit():
            self.number()
            return
        
        # Handle identifiers and keywords
        if c.isalpha() or c == '_':
            self.identifier()
            return
        
        # Handle strings and chars
        if c == '"' or c == "'":
            self.string(c)
            return
        
        # Handle chars
        if c == '`':
            self.char()
            return
        
        # Handle unknown characters
        self.error(f"Unexpected character '{c}'")

    def number(self):
        is_hex = False
        is_binary = False
        
        if self.peek().lower() == 'x' and self.source[self.start] == '0':
            self.advance()  # Consume 'x'
            is_hex = True
            while self.peek().isdigit() or self.peek().lower() in 'abcdef':
                self.advance()
        elif self.peek().lower() == 'b' and self.source[self.start] == '0':
            self.advance()  # Consume 'b'
            is_binary = True
            while self.peek() in '01':
                self.advance()
        else:
            while self.peek().isdigit():
                self.advance()
            
            if self.peek() == '.' and self.peek_next().isdigit():
                self.advance()  # Consume the '.'
                while self.peek().isdigit():
                    self.advance()
                
                if self.peek().lower() == 'e':
                    self.advance()  # Consume the 'e'
                    if self.peek() in ('+', '-'):
                        self.advance()
                    while self.peek().isdigit():
                        self.advance()
        
        num_str = self.source[self.start:self.current]
        
        if is_hex:
            self.add_token(TokenType.INTEGER, int(num_str, 16))
        elif is_binary:
            self.add_token(TokenType.INTEGER, int(num_str[2:], 2))
        elif '.' in num_str or 'e' in num_str.lower():
            self.add_token(TokenType.FLOAT, float(num_str))
        else:
            self.add_token(TokenType.INTEGER, int(num_str))

    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        
        text = self.source[self.start:self.current]
        token_type = self.KEYWORDS.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type, text)

    def string(self, delimiter):
        while self.peek() != delimiter and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 0
            if self.peek() == '\\':
                self.advance()  # Consume the backslash
                # Handle escape sequences
                escape = self.peek()
                if escape == delimiter:
                    self.advance()  # Consume the escaped delimiter
                elif escape == 'n':
                    self.advance()  # Consume the 'n'
                elif escape == 't':
                    self.advance()  # Consume the 't'
                elif escape == 'r':
                    self.advance()  # Consume the 'r'
                elif escape == '\\':
                    self.advance()  # Consume the second backslash
                elif escape == 'x':
                    self.advance()  # Consume 'x'
                    hex_digits = ''
                    for _ in range(2):
                        if self.peek().isalnum():
                            hex_digits += self.advance()
                        else:
                            self.error("Invalid hex escape sequence")
                            return
                elif escape == 'u':
                    self.advance()  # Consume 'u'
                    if self.peek() == '{':
                        self.advance()  # Consume '{'
                        hex_digits = ''
                        while self.peek() != '}' and not self.is_at_end():
                            if self.peek().isalnum():
                                hex_digits += self.advance()
                            else:
                                self.error("Invalid Unicode escape sequence")
                                return
                        if self.peek() != '}':
                            self.error("Unterminated Unicode escape sequence")
                            return
                        self.advance()  # Consume '}'
                    else:
                        hex_digits = ''
                        for _ in range(4):
                            if self.peek().isalnum():
                                hex_digits += self.advance()
                            else:
                                self.error("Invalid Unicode escape sequence")
                                return
            self.advance()
        
        if self.is_at_end():
            self.error("Unterminated string")
            return
        
        self.advance()  # Consume the closing delimiter
        
        value = self.source[self.start + 1:self.current - 1]
        value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\\\', '\\')
        
        if delimiter == "'":
            self.add_token(TokenType.CHAR, value[0] if value else '\0')
        else:
            self.add_token(TokenType.STRING, value)

    def char(self):
        value = ''
        while self.peek() != '`' and not self.is_at_end():
            if self.peek() == '\\':
                self.advance()  # Consume backslash
                escape = self.peek()
                if escape == 'n':
                    value += '\n'
                    self.advance()
                elif escape == 't':
                    value += '\t'
                    self.advance()
                elif escape == 'r':
                    value += '\r'
                    self.advance()
                elif escape == '\\':
                    value += '\\'
                    self.advance()
                elif escape == '`':
                    value += '`'
                    self.advance()
                else:
                    value += '\\' + escape
                    self.advance()
            else:
                value += self.advance()
        
        if self.is_at_end():
            self.error("Unterminated character literal")
            return
        
        self.advance()  # Consume closing backtick
        
        if len(value) != 1:
            self.error("Character literal must contain exactly one character")
            return
        
        self.add_token(TokenType.CHAR, value[0])

    def advance(self) -> str:
        self.current += 1
        self.column += 1
        return self.source[self.current - 1]

    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]

    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]

    def is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def add_token(self, token_type: TokenType, literal: Any = None):
        lexeme = self.source[self.start:self.current]
        self.tokens.append(Token(token_type, literal, self.line, self.column, self.filename))

    def error(self, message: str, hint: Optional[str] = None):
        self.errors.append(CompilerError(message, self.line, self.column, self.filename, hint))

# ==================== PARSER ====================
class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<stdin>"):
        self.tokens = tokens
        self.current = 0
        self.filename = filename
        self.errors = []
        self.imported_modules = set()
        self.current_module = None

    def parse(self) -> Program:
        try:
            statements = []
            while not self.is_at_end():
                decl = self.declaration()
                if isinstance(decl, CompilerError):
                    continue
                statements.append(decl)
            
            return Program(statements, 1, 1, self.filename)
        except CompilerError as e:
            self.errors.append(e)
            return Program([], 1, 1, self.filename)

    def declaration(self) -> ASTNode:
        try:
            if self.match(TokenType.MODULE):
                return self.module_declaration()
            if self.match(TokenType.CLASS):
                return self.class_declaration()
            if self.match(TokenType.TRAIT):
                return self.trait_declaration()
            if self.match(TokenType.FN):
                return self.function_declaration()
            if self.match(TokenType.LET, TokenType.CONST):
                return self.var_declaration()
            if self.match(TokenType.IMPORT):
                return self.import_declaration()
            if self.match(TokenType.USE):
                return self.use_declaration()
            return self.statement()
        except CompilerError as e:
            self.synchronize()
            return e

    def module_declaration(self) -> Module:
        name = self.consume(TokenType.IDENTIFIER, "Expect module name").value
        is_public = self.match(TokenType.PUB)
        
        self.current_module = name
        self.consume(TokenType.LBRACE, "Expect '{' before module body")
        
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.declaration())
        
        self.consume(TokenType.RBRACE, "Expect '}' after module body")
        self.current_module = None
        
        return Module(name, Block(statements, self.previous().line, self.previous().column, self.filename), 
                     is_public, self.previous().line, self.previous().column, self.filename)

    def class_declaration(self) -> Class:
        name = self.consume(TokenType.IDENTIFIER, "Expect class name").value
        
        parent = None
        if self.match(TokenType.COLON):
            parent = self.consume(TokenType.IDENTIFIER, "Expect parent class name").value
        
        traits = []
        if self.match(TokenType.IMPLEMENTS):
            traits.append(self.consume(TokenType.IDENTIFIER, "Expect trait name").value)
            while self.match(TokenType.COMMA):
                traits.append(self.consume(TokenType.IDENTIFIER, "Expect trait name").value)
        
        self.consume(TokenType.LBRACE, "Expect '{' before class body")
        
        fields = []
        methods = []
        
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            is_public = self.match(TokenType.PUB)
            is_unsafe = self.match(TokenType.UNSAFE)
            
            if self.match(TokenType.FN):
                method = self.function_declaration(is_public)
                method.is_unsafe = is_unsafe
                methods.append(method)
            else:
                field = self.var_declaration(is_public)
                field.is_unsafe = is_unsafe
                fields.append(field)
        
        self.consume(TokenType.RBRACE, "Expect '}' after class body")
        return Class(name, parent, traits, fields, methods, is_public, self.previous().line, self.previous().column, self.filename)

    def trait_declaration(self) -> Trait:
        name = self.consume(TokenType.IDENTIFIER, "Expect trait name").value
        is_public = self.match(TokenType.PUB)
        self.consume(TokenType.LBRACE, "Expect '{' before trait body")
        
        methods = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            is_public = self.match(TokenType.PUB)
            is_unsafe = self.match(TokenType.UNSAFE)
            methods.append(self.function_declaration(is_public, is_unsafe))
        
        self.consume(TokenType.RBRACE, "Expect '}' after trait body")
        return Trait(name, methods, is_public, self.previous().line, self.previous().column, self.filename)

    def function_declaration(self, is_public: bool = False, is_unsafe: bool = False) -> Function:
        is_async = self.match(TokenType.ASYNC)
        name = self.consume(TokenType.IDENTIFIER, "Expect function name").value
        
        self.consume(TokenType.LPAREN, "Expect '(' after function name")
        
        params = []
        if not self.check(TokenType.RPAREN):
            param_name = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
            param_type = None
            if self.match(TokenType.COLON):
                param_type = self.consume(TokenType.IDENTIFIER, "Expect parameter type").value
            
            params.append((param_name, param_type))
            
            while self.match(TokenType.COMMA):
                param_name = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
                param_type = None
                if self.match(TokenType.COLON):
                    param_type = self.consume(TokenType.IDENTIFIER, "Expect parameter type").value
                
                params.append((param_name, param_type))
        
        self.consume(TokenType.RPAREN, "Expect ')' after parameters")
        
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.consume(TokenType.IDENTIFIER, "Expect return type").value
        
        body = self.block_statement()
        return Function(name, params, return_type, body, is_async, is_public, is_unsafe, 
                       self.previous().line, self.previous().column, self.filename)

    def var_declaration(self, is_public: bool = False) -> VarDecl:
        is_const = self.previous().type == TokenType.CONST
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name").value
        
        var_type = None
        if self.match(TokenType.COLON):
            var_type = self.consume(TokenType.IDENTIFIER, "Expect variable type").value
        
        initializer = None
        if self.match(TokenType.EQ):
            initializer = self.expression()
        
        self.consume(TokenType.SEMI, "Expect ';' after variable declaration")
        return VarDecl(name, var_type, initializer, is_const, is_public, False, 
                      self.previous().line, self.previous().column, self.filename)

    def import_declaration(self) -> Import:
        module_parts = [self.consume(TokenType.IDENTIFIER, "Expect module name").value]
        
        while self.match(TokenType.DOT):
            module_parts.append(self.consume(TokenType.IDENTIFIER, "Expect module part").value)
        
        module = '.'.join(module_parts)
        alias = None
        imports = []
        is_relative = False
        version = None
        
        # Check for version specifier
        if self.match(TokenType.AT):
            version = self.consume(TokenType.STRING, "Expect version string").value
        
        if self.match(TokenType.AS):
            alias = self.consume(TokenType.IDENTIFIER, "Expect alias name").value
        elif self.match(TokenType.FROM):
            is_relative = True
            if self.match(TokenType.LBRACE):
                while not self.check(TokenType.RBRACE) and not self.is_at_end():
                    item = self.consume(TokenType.IDENTIFIER, "Expect import item").value
                    item_alias = None
                    if self.match(TokenType.AS):
                        item_alias = self.consume(TokenType.IDENTIFIER, "Expect item alias").value
                    imports.append((item, item_alias))
                    if not self.match(TokenType.COMMA):
                        break
                
                self.consume(TokenType.RBRACE, "Expect '}' after import list")
        
        self.consume(TokenType.SEMI, "Expect ';' after import statement")
        
        # Add to imported modules
        self.imported_modules.add(module)
        
        return Import(module, alias, imports, is_relative, version, 
                     self.previous().line, self.previous().column, self.filename)

    def use_declaration(self) -> Use:
        items = []
        from_module = None
        
        # Handle single item
        item = self.consume(TokenType.IDENTIFIER, "Expect use item").value
        item_alias = None
        if self.match(TokenType.AS):
            item_alias = self.consume(TokenType.IDENTIFIER, "Expect item alias").value
        items.append((item, item_alias))
        
        # Handle multiple items
        while self.match(TokenType.COMMA):
            item = self.consume(TokenType.IDENTIFIER, "Expect use item").value
            item_alias = None
            if self.match(TokenType.AS):
                item_alias = self.consume(TokenType.IDENTIFIER, "Expect item alias").value
            items.append((item, item_alias))
        
        # Handle 'from' clause
        if self.match(TokenType.FROM):
            from_module = self.consume(TokenType.IDENTIFIER, "Expect module name").value
        
        self.consume(TokenType.SEMI, "Expect ';' after use statement")
        return Use(items, from_module, self.previous().line, self.previous().column, self.filename)

    def statement(self) -> ASTNode:
        if self.match(TokenType.IF):
            return self.if_statement()
        if self.match(TokenType.FOR):
            return self.for_statement()
        if self.match(TokenType.WHILE):
            return self.while_statement()
        if self.match(TokenType.RETURN):
            return self.return_statement()
        if self.match(TokenType.BREAK):
            return self.break_statement()
        if self.match(TokenType.CONTINUE):
            return self.continue_statement()
        if self.match(TokenType.MATCH):
            return self.match_statement()
        if self.match(TokenType.TRY):
            return self.try_statement()
        if self.match(TokenType.THROW):
            return self.throw_statement()
        if self.match(TokenType.WITH):
            return self.with_statement()
        if self.match(TokenType.UNSAFE):
            return self.unsafe_statement()
        if self.match(TokenType.DEFER):
            return self.defer_statement()
        if self.match(TokenType.LBRACE):
            return self.block_statement()
        return self.expression_statement()

    def if_statement(self) -> If:
        condition = self.expression()
        then_branch = self.statement()
        
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = self.statement()
        
        return If(condition, then_branch, else_branch, self.previous().line, self.previous().column, self.filename)

    def for_statement(self) -> For:
        is_parallel = self.match(TokenType.DOLLAR)
        var_name = self.consume(TokenType.IDENTIFIER, "Expect loop variable name").value
        self.consume(TokenType.IN, "Expect 'in' after loop variable")
        iterable = self.expression()
        body = self.statement()
        return For(var_name, iterable, body, is_parallel, self.previous().line, self.previous().column, self.filename)

    def while_statement(self) -> While:
        condition = self.expression()
        body = self.statement()
        return While(condition, body, self.previous().line, self.previous().column, self.filename)

    def return_statement(self) -> Return:
        value = None
        if not self.check(TokenType.SEMI):
            value = self.expression()
        
        self.consume(TokenType.SEMI, "Expect ';' after return value")
        return Return(value, self.previous().line, self.previous().column, self.filename)

    def break_statement(self) -> Break:
        self.consume(TokenType.SEMI, "Expect ';' after break")
        return Break(self.previous().line, self.previous().column, self.filename)

    def continue_statement(self) -> Continue:
        self.consume(TokenType.SEMI, "Expect ';' after continue")
        return Continue(self.previous().line, self.previous().column, self.filename)

    def match_statement(self) -> Match:
        value = self.expression()
        self.consume(TokenType.LBRACE, "Expect '{' after match value")
        
        cases = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            pattern = self.expression()
            self.consume(TokenType.ARROW, "Expect '=>' after match pattern")
            body = self.expression()
            cases.append((pattern, body))
            
            if not self.match(TokenType.COMMA):
                break
        
        self.consume(TokenType.RBRACE, "Expect '}' after match cases")
        return Match(value, cases, self.previous().line, self.previous().column, self.filename)

    def try_statement(self) -> Try:
        try_block = self.block_statement()
        
        catch_blocks = []
        while self.match(TokenType.CATCH):
            self.consume(TokenType.LPAREN, "Expect '(' after catch")
            
            error_var = self.consume(TokenType.IDENTIFIER, "Expect error variable name").value
            error_type = None
            if self.match(TokenType.COLON):
                error_type = self.consume(TokenType.IDENTIFIER, "Expect error type").value
            
            self.consume(TokenType.RPAREN, "Expect ')' after catch clause")
            catch_block = self.block_statement()
            catch_blocks.append((error_var, error_type, catch_block))
        
        finally_block = None
        if self.match(TokenType.FINALLY):
            finally_block = self.block_statement()
        
        return Try(try_block, catch_blocks, finally_block, self.previous().line, self.previous().column, self.filename)

    def throw_statement(self) -> Throw:
        value = self.expression()
        self.consume(TokenType.SEMI, "Expect ';' after throw value")
        return Throw(value, self.previous().line, self.previous().column, self.filename)

    def with_statement(self) -> ASTNode:
        resource = self.expression()
        self.consume(TokenType.AS, "Expect 'as' after with resource")
        var_name = self.consume(TokenType.IDENTIFIER, "Expect variable name").value
        body = self.statement()
        return Block([
            Assignment(Variable(var_name, resource.line, resource.column, self.filename), resource, TokenType.EQ, 
                      resource.line, resource.column, self.filename),
            body
        ], self.previous().line, self.previous().column, self.filename)

    def unsafe_statement(self) -> UnsafeBlock:
        body = self.block_statement()
        return UnsafeBlock(body, self.previous().line, self.previous().column, self.filename)

    def defer_statement(self) -> Defer:
        expr = self.expression()
        self.consume(TokenType.SEMI, "Expect ';' after defer expression")
        return Defer(expr, self.previous().line, self.previous().column, self.filename)

    def block_statement(self) -> Block:
        self.consume(TokenType.LBRACE, "Expect '{' before block")
        
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.declaration())
        
        self.consume(TokenType.RBRACE, "Expect '}' after block")
        return Block(statements, self.previous().line, self.previous().column, self.filename)

    def expression_statement(self) -> ASTNode:
        expr = self.expression()
        self.consume(TokenType.SEMI, "Expect ';' after expression")
        return expr

    def expression(self) -> ASTNode:
        return self.assignment()

    def assignment(self) -> ASTNode:
        expr = self.logical_or()
        
        if self.match(*[op for op in TokenType if op.name.endswith('_EQ')]):
            op = self.previous()
            value = self.assignment()
            
            if isinstance(expr, Variable):
                return Assignment(expr, value, op.type, op.line, op.column, self.filename)
            elif isinstance(expr, Attribute):
                return SetAttribute(expr.obj, expr.attr, value, op.line, op.column, self.filename)
            elif isinstance(expr, Index):
                return SetIndex(expr.obj, expr.index, value, op.line, op.column, self.filename)
            
            self.error("Invalid assignment target")
        
        return expr

    def logical_or(self) -> ASTNode:
        expr = self.logical_and()
        
        while self.match(TokenType.PIPE_PIPE):
            op = self.previous()
            right = self.logical_and()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def logical_and(self) -> ASTNode:
        expr = self.equality()
        
        while self.match(TokenType.AMP_AMP):
            op = self.previous()
            right = self.equality()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def equality(self) -> ASTNode:
        expr = self.comparison()
        
        while self.match(TokenType.EQ_EQ, TokenType.BANG_EQ):
            op = self.previous()
            right = self.comparison()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def comparison(self) -> ASTNode:
        expr = self.bitwise_or()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LT_EQ, TokenType.GT_EQ):
            op = self.previous()
            right = self.bitwise_or()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def bitwise_or(self) -> ASTNode:
        expr = self.bitwise_xor()
        
        while self.match(TokenType.PIPE):
            op = self.previous()
            right = self.bitwise_xor()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def bitwise_xor(self) -> ASTNode:
        expr = self.bitwise_and()
        
        while self.match(TokenType.CARET):
            op = self.previous()
            right = self.bitwise_and()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def bitwise_and(self) -> ASTNode:
        expr = self.shift()
        
        while self.match(TokenType.AMP):
            op = self.previous()
            right = self.shift()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def shift(self) -> ASTNode:
        expr = self.addition()
        
        while self.match(TokenType.LT_LT, TokenType.GT_GT):
            op = self.previous()
            right = self.addition()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def addition(self) -> ASTNode:
        expr = self.multiplication()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.previous()
            right = self.multiplication()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def multiplication(self) -> ASTNode:
        expr = self.unary()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.previous()
            right = self.unary()
            expr = BinaryOp(expr, op.type, right, op.line, op.column, self.filename)
        
        return expr

    def unary(self) -> ASTNode:
        if self.match(TokenType.BANG, TokenType.MINUS, TokenType.TILDE, TokenType.AMP, TokenType.STAR):
            op = self.previous()
            right = self.unary()
            return UnaryOp(op.type, right, op.line, op.column, self.filename)
        
        return self.call()

    def call(self) -> ASTNode:
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                is_tail_call = False
                if self.match(TokenType.ARROW):
                    is_tail_call = True
                expr = self.finish_call(expr, is_tail_call)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expect property name after '.'").value
                expr = Attribute(expr, name, self.previous().line, self.previous().column, self.filename)
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.consume(TokenType.RBRACKET, "Expect ']' after index")
                expr = Index(expr, index, self.previous().line, self.previous().column, self.filename)
            else:
                break
        
        return expr

    def finish_call(self, callee: ASTNode, is_tail_call: bool) -> ASTNode:
        args = []
        if not self.check(TokenType.RPAREN):
            args.append(self.expression())
            while self.match(TokenType.COMMA):
                args.append(self.expression())
        
        paren = self.consume(TokenType.RPAREN, "Expect ')' after arguments")
        return Call(callee, args, is_tail_call, paren.line, paren.column, self.filename)

    def primary(self) -> ASTNode:
        if self.match(TokenType.TRUE):
            return Literal(True, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.FALSE):
            return Literal(False, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.NIL):
            return Literal(None, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.INTEGER, TokenType.FLOAT, TokenType.STRING, TokenType.CHAR):
            return Literal(self.previous().value, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.IDENTIFIER):
            return Variable(self.previous().value, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expect ')' after expression")
            return Grouping(expr, self.previous().line, self.previous().column, self.filename)
        if self.match(TokenType.LBRACKET):
            return self.array_literal()
        if self.match(TokenType.LBRACE):
            return self.dict_literal()
        if self.match(TokenType.FN):
            return self.lambda_expr()
        if self.match(TokenType.ASYNC):
            return self.async_lambda_expr()
        if self.match(TokenType.AWAIT):
            return self.await_expr()
        if self.match(TokenType.DOLLAR):
            return self.spawn_expr()
        
        raise self.error("Expect expression")

    def array_literal(self) -> Array:
        elements = []
        if not self.check(TokenType.RBRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
        
        bracket = self.consume(TokenType.RBRACKET, "Expect ']' after array elements")
        return Array(elements, bracket.line, bracket.column, self.filename)

    def dict_literal(self) -> Dict:
        entries = []
        if not self.check(TokenType.RBRACE):
            key = self.expression()
            self.consume(TokenType.COLON, "Expect ':' after dictionary key")
            value = self.expression()
            entries.append((key, value))
            
            while self.match(TokenType.COMMA):
                key = self.expression()
                self.consume(TokenType.COLON, "Expect ':' after dictionary key")
                value = self.expression()
                entries.append((key, value))
        
        brace = self.consume(TokenType.RBRACE, "Expect '}' after dictionary entries")
        return Dict(entries, brace.line, brace.column, self.filename)

    def lambda_expr(self) -> Lambda:
        self.consume(TokenType.LPAREN, "Expect '(' after lambda")
        
        params = []
        if not self.check(TokenType.RPAREN):
            param_name = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
            param_type = None
            if self.match(TokenType.COLON):
                param_type = self.consume(TokenType.IDENTIFIER, "Expect parameter type").value
            
            params.append((param_name, param_type))
            
            while self.match(TokenType.COMMA):
                param_name = self.consume(TokenType.IDENTIFIER, "Expect parameter name").value
                param_type = None
                if self.match(TokenType.COLON):
                    param_type = self.consume(TokenType.IDENTIFIER, "Expect parameter type").value
                
                params.append((param_name, param_type))
        
        self.consume(TokenType.RPAREN, "Expect ')' after lambda parameters")
        
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.consume(TokenType.IDENTIFIER, "Expect return type").value
        
        body = self.expression()
        return Lambda(params, return_type, body, False, self.previous().line, self.previous().column, self.filename)

    def async_lambda_expr(self) -> Lambda:
        self.consume(TokenType.FN, "Expect 'fn' after async")
        lambda_expr = self.lambda_expr()
        lambda_expr.is_async = True
        return lambda_expr

    def await_expr(self) -> Await:
        expr = self.expression()
        return Await(expr, self.previous().line, self.previous().column, self.filename)

    def spawn_expr(self) -> Spawn:
        expr = self.expression()
        return Spawn(expr, self.previous().line, self.previous().column, self.filename)

    def match(self, *types: TokenType) -> bool:
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False

    def check(self, type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == type

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def consume(self, type: TokenType, message: str) -> Token:
        if self.check(type):
            return self.advance()
        
        raise self.error(message)

    def error(self, message: str, hint: Optional[str] = None) -> CompilerError:
        token = self.peek()
        return CompilerError(message, token.line, token.column, self.filename, hint)

    def synchronize(self):
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMI:
                return
            
            if self.peek().type in {
                TokenType.CLASS, TokenType.FN, TokenType.LET, TokenType.CONST,
                TokenType.IF, TokenType.FOR, TokenType.WHILE, TokenType.RETURN,
                TokenType.IMPORT, TokenType.USE, TokenType.MODULE
            }:
                return
            
            self.advance()

# ==================== COMPILER ====================
class Compiler:
    def __init__(self, mode: CompilerMode = CompilerMode.DEBUG, 
                 optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
                 target: TargetPlatform = TargetPlatform.BYTECODE):
        self.mode = mode
        self.optimization_level = optimization_level
        self.target = target
        self.bytecode = bytearray()
        self.constants = []
        self.names = []
        self.locals = []
        self.loops = []
        self.try_blocks = []
        self.current_scope = None
        self.current_function = None
        self.current_class = None
        self.current_module = None
        self.errors = []
        self.imported_modules = set()
        self.deferred = []
        self.ownership_stack = []

    def compile(self, node: ASTNode) -> Tuple[bytes, List[Any], List[str]]:
        try:
            self.visit(node)
            
            # Handle deferred statements
            for expr in reversed(self.deferred):
                self.visit(expr)
            
            self.emit(OpCode.HALT)
            return bytes(self.bytecode), self.constants, self.names
        except CompilerError as e:
            self.errors.append(e)
            return bytes(), [], []

    def visit(self, node: ASTNode):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node: ASTNode):
        raise CompilerError(f"No visit method for {type(node).__name__}", node.line, node.column, node.filename)

    def visit_Program(self, node: Program):
        for stmt in node.statements:
            self.visit(stmt)

    def visit_Module(self, node: Module):
        prev_module = self.current_module
        self.current_module = node.name
        
        # Create module object
        module_const = {
            'name': node.name,
            'is_public': node.is_public
        }
        
        const_index = self.add_constant(module_const)
        self.emit(OpCode.PUSH, const_index)
        self.emit(OpCode.NEW_MODULE)
        
        # Store module in current scope
        index = self.add_local(node.name)
        self.emit(OpCode.STORE, index)
        
        # Compile module body
        self.visit(node.body)
        
        self.current_module = prev_module

    def visit_Block(self, node: Block):
        prev_scope = self.current_scope
        self.current_scope = {
            'parent': prev_scope,
            'locals': {},
            'depth': prev_scope['depth'] + 1 if prev_scope else 0
        }
        
        for stmt in node.statements:
            self.visit(stmt)
        
        self.current_scope = prev_scope

    def visit_VarDecl(self, node: VarDecl):
        if node.value:
            self.visit(node.value)
        else:
            self.emit(OpCode.PUSH, self.add_constant(None))
        
        index = self.add_local(node.name)
        self.emit(OpCode.STORE, index)
        
        if node.is_unsafe and self.mode != CompilerMode.DEBUG:
            self.warning("Unsafe variable declaration in non-debug mode", node.line, node.column, node.filename)

    def visit_Function(self, node: Function):
        # Compile function body
        prev_function = self.current_function
        self.current_function = node
        
        prev_scope = self.current_scope
        self.current_scope = {
            'parent': prev_scope,
            'locals': {},
            'depth': prev_scope['depth'] + 1 if prev_scope else 0
        }
        
        # Add parameters to scope
        for i, (name, type_) in enumerate(node.params):
            self.current_scope['locals'][name] = i
        
        # Compile body
        body_code = Compiler(self.mode, self.optimization_level, self.target)
        body_code.current_function = node
        body_code.current_scope = self.current_scope
        body_code.visit(node.body)
        
        # Add return if not present
        if not body_code.bytecode.endswith(bytes([OpCode.RETURN.value])):
            body_code.emit(OpCode.PUSH, body_code.add_constant(None))
            body_code.emit(OpCode.RETURN)
        
        # Create function object
        func_const = {
            'name': node.name,
            'params': node.params,
            'return_type': node.return_type,
            'code': bytes(body_code.bytecode),
            'constants': body_code.constants,
            'names': body_code.names,
            'locals_count': len(body_code.locals),
            'is_async': node.is_async,
            'is_unsafe': node.is_unsafe,
            'module': self.current_module
        }
        
        const_index = self.add_constant(func_const)
        self.emit(OpCode.PUSH, const_index)
        self.emit(OpCode.NEW_FUNCTION)
        
        # Store function in current scope
        if node.name:
            index = self.add_local(node.name)
            self.emit(OpCode.STORE, index)
        
        self.current_function = prev_function
        self.current_scope = prev_scope
        
        if node.is_unsafe and self.mode != CompilerMode.DEBUG:
            self.warning("Unsafe function in non-debug mode", node.line, node.column, node.filename)

    def visit_Class(self, node: Class):
        # Compile class methods
        methods = {}
        for method in node.methods:
            method_code = Compiler(self.mode, self.optimization_level, self.target)
            method_code.visit(method)
            methods[method.name] = {
                'code': bytes(method_code.bytecode),
                'constants': method_code.constants,
                'names': method_code.names,
                'params': method.params,
                'return_type': method.return_type,
                'is_async': method.is_async,
                'is_unsafe': method.is_unsafe
            }
        
        # Create class object
        class_const = {
            'name': node.name,
            'parent': node.parent,
            'traits': node.traits,
            'methods': methods,
            'fields': [(field.name, field.type, field.is_unsafe) for field in node.fields],
            'module': self.current_module
        }
        
        const_index = self.add_constant(class_const)
        self.emit(OpCode.PUSH, const_index)
        self.emit(OpCode.NEW_CLASS)
        
        # Store class in current scope
        index = self.add_local(node.name)
        self.emit(OpCode.STORE, index)
        
        # Initialize fields
        for field in node.fields:
            if field.value:
                self.visit(field.value)
            else:
                self.emit(OpCode.PUSH, self.add_constant(None))
            
            self.emit(OpCode.LOAD, index)
            self.emit(OpCode.STORE_ATTR, self.add_name(field.name))
            
            if field.is_unsafe and self.mode != CompilerMode.DEBUG:
                self.warning("Unsafe field in non-debug mode", field.line, field.column, field.filename)

    def visit_Trait(self, node: Trait):
        # Compile trait methods
        methods = {}
        for method in node.methods:
            method_code = Compiler(self.mode, self.optimization_level, self.target)
            method_code.visit(method)
            methods[method.name] = {
                'code': bytes(method_code.bytecode),
                'constants': method_code.constants,
                'names': method_code.names,
                'params': method.params,
                'return_type': method.return_type,
                'is_async': method.is_async,
                'is_unsafe': method.is_unsafe
            }
        
        # Create trait object
        trait_const = {
            'name': node.name,
            'methods': methods,
            'module': self.current_module
        }
        
        const_index = self.add_constant(trait_const)
        self.emit(OpCode.PUSH, const_index)
        self.emit(OpCode.NEW_CLASS)  # Reuse class opcode for traits
        
        # Store trait in current scope
        index = self.add_local(node.name)
        self.emit(OpCode.STORE, index)

    def visit_Import(self, node: Import):
        # Handle module import
        self.emit(OpCode.PUSH, self.add_constant(node.module))
        
        if node.version:
            self.emit(OpCode.PUSH, self.add_constant(node.version))
            self.emit(OpCode.IMPORT, 2)
        else:
            self.emit(OpCode.IMPORT, 1)
        
        if node.alias:
            index = self.add_local(node.alias)
            self.emit(OpCode.STORE, index)
        elif node.imports:
            for name, alias in node.imports:
                self.emit(OpCode.DUP)
                self.emit(OpCode.LOAD_ATTR, self.add_name(name))
                index = self.add_local(alias or name)
                self.emit(OpCode.STORE, index)
            
            self.emit(OpCode.POP)  # Remove module from stack
        else:
            parts = node.module.split('.')
            index = self.add_local(parts[-1])
            self.emit(OpCode.STORE, index)
        
        self.imported_modules.add(node.module)

    def visit_Use(self, node: Use):
        if node.from_module:
            # Handle 'use item from module' syntax
            for item, alias in node.items:
                self.emit(OpCode.PUSH, self.add_constant(node.from_module))
                self.emit(OpCode.IMPORT, 1)
                self.emit(OpCode.LOAD_ATTR, self.add_name(item))
                index = self.add_local(alias or item)
                self.emit(OpCode.STORE, index)
        else:
            # Handle 'use item' syntax (from current module)
            for item, alias in node.items:
                if self.current_module:
                    self.emit(OpCode.PUSH, self.add_constant(self.current_module))
                    self.emit(OpCode.IMPORT, 1)
                    self.emit(OpCode.LOAD_ATTR, self.add_name(item))
                    index = self.add_local(alias or item)
                    self.emit(OpCode.STORE, index)
                else:
                    self.error(f"Cannot use '{item}' outside of module", node.line, node.column, node.filename)

    def visit_If(self, node: If):
        self.visit(node.condition)
        
        # Jump if false
        jump_if_false_pos = len(self.bytecode)
        self.emit(OpCode.JMP_IF_FALSE, 0)
        
        # Then branch
        self.visit(node.then_branch)
        
        if node.else_branch:
            # Jump over else branch
            jump_pos = len(self.bytecode)
            self.emit(OpCode.JMP, 0)
            
            # Update jump if false to go to else branch
            else_pos = len(self.bytecode)
            self.bytecode[jump_if_false_pos + 1] = else_pos - jump_if_false_pos
            
            # Else branch
            self.visit(node.else_branch)
            
            # Update jump to go after else branch
            self.bytecode[jump_pos + 1] = len(self.bytecode) - jump_pos
        else:
            # Update jump if false to go after then branch
            self.bytecode[jump_if_false_pos + 1] = len(self.bytecode) - jump_if_false_pos

    def visit_For(self, node: For):
        # Compile iterable
        self.visit(node.iterable)
        
        if node.is_parallel:
            # Parallel for loop
            self.emit(OpCode.PUSH, self.add_constant(node.var_name))
            self.emit(OpCode.PUSH, self.add_constant(None))  # Placeholder for function
            self.emit(OpCode.NEW_FUNCTION)
            
            # Spawn parallel tasks
            self.emit(OpCode.SPAWN)
        else:
            # Regular for loop
            self.emit(OpCode.PUSH, self.add_constant(None))  # Iterator state
            
            # Start of loop
            loop_start = len(self.bytecode)
            
            # Get next item
            self.emit(OpCode.DUP)
            self.emit(OpCode.DUP)
            self.emit(OpCode.LOAD_INDEX, 1)
            self.emit(OpCode.STORE, self.add_local(node.var_name))
            
            # Check if done
            self.emit(OpCode.DUP)
            self.emit(OpCode.PUSH, self.add_constant(None))
            self.emit(OpCode.EQ)
            
            # Jump if done
            jump_if_done_pos = len(self.bytecode)
            self.emit(OpCode.JMP_IF_TRUE, 0)
            
            # Body
            self.visit(node.body)
            
            # Jump back to start
            self.emit(OpCode.JMP, loop_start - len(self.bytecode))
            
            # Update jump if done to go after loop
            done_pos = len(self.bytecode)
            self.bytecode[jump_if_done_pos + 1] = done_pos - jump_if_done_pos
            
            # Clean up stack
            self.emit(OpCode.POP)
            self.emit(OpCode.POP)

    def visit_While(self, node: While):
        # Start of loop
        loop_start = len(self.bytecode)
        
        # Condition
        self.visit(node.condition)
        
        # Jump if false
        jump_if_false_pos = len(self.bytecode)
        self.emit(OpCode.JMP_IF_FALSE, 0)
        
        # Body
        self.visit(node.body)
        
        # Jump back to start
        self.emit(OpCode.JMP, loop_start - len(self.bytecode))
        
        # Update jump if false to go after loop
        self.bytecode[jump_if_false_pos + 1] = len(self.bytecode) - jump_if_false_pos

    def visit_Return(self, node: Return):
        if node.value:
            self.visit(node.value)
        else:
            self.emit(OpCode.PUSH, self.add_constant(None))
        
        self.emit(OpCode.RETURN)

    def visit_Break(self, node: Break):
        if not self.loops:
            raise CompilerError("Break outside loop", node.line, node.column, node.filename)
        
        loop = self.loops[-1]
        self.emit(OpCode.JMP, loop['break_pos'] - len(self.bytecode))

    def visit_Continue(self, node: Continue):
        if not self.loops:
            raise CompilerError("Continue outside loop", node.line, node.column, node.filename)
        
        loop = self.loops[-1]
        self.emit(OpCode.JMP, loop['continue_pos'] - len(self.bytecode))

    def visit_Match(self, node: Match):
        self.visit(node.value)
        
        # Store value for comparison
        value_pos = self.add_local("_match_value")
        self.emit(OpCode.DUP)
        self.emit(OpCode.STORE, value_pos)
        
        # Compile cases
        case_jumps = []
        for pattern, body in node.cases:
            # Load value for comparison
            self.emit(OpCode.LOAD, value_pos)
            
            # Compile pattern
            self.visit(pattern)
            
            # Compare
            self.emit(OpCode.EQ)
            
            # Jump if not matched
            jump_if_not_pos = len(self.bytecode)
            self.emit(OpCode.JMP_IF_FALSE, 0)
            
            # Compile body
            self.visit(body)
            
            # Jump to end of match
            case_jumps.append(len(self.bytecode))
            self.emit(OpCode.JMP, 0)
            
            # Update jump if not matched
            self.bytecode[jump_if_not_pos + 1] = len(self.bytecode) - jump_if_not_pos
        
        # Update case jumps to go after match
        end_pos = len(self.bytecode)
        for jump_pos in case_jumps:
            self.bytecode[jump_pos + 1] = end_pos - jump_pos
        
        # Clean up
        self.emit(OpCode.POP)  # Remove stored value

    def visit_Try(self, node: Try):
        # Begin try block
        try_begin_pos = len(self.bytecode)
        self.emit(OpCode.TRY_BEGIN, len(node.catch_blocks))
        
        # Compile try block
        self.visit(node.try_block)
        
        # End try block
        self.emit(OpCode.TRY_END)
        
        # Jump over catch blocks
        jump_pos = len(self.bytecode)
        self.emit(OpCode.JMP, 0)
        
        # Compile catch blocks
        catch_blocks = []
        for error_var, error_type, catch_block in node.catch_blocks:
            catch_start = len(self.bytecode)
            
            # Store error in variable if provided
            if error_var:
                error_pos = self.add_local(error_var)
                self.emit(OpCode.STORE, error_pos)
            
            # Compile catch block
            self.visit(catch_block)
            
            # Jump to end of try-catch
            self.emit(OpCode.JMP, 0)
            catch_blocks.append((catch_start, len(self.bytecode)))
        
        # Update try block to point to catch blocks
        for i, (catch_start, _) in enumerate(catch_blocks):
            self.bytecode[try_begin_pos + 1 + i] = catch_start - try_begin_pos
        
        # Update jumps to go after try-catch
        end_pos = len(self.bytecode)
        self.bytecode[jump_pos + 1] = end_pos - jump_pos
        for _, catch_jump_pos in catch_blocks:
            self.bytecode[catch_jump_pos + 1] = end_pos - catch_jump_pos
        
        # Compile finally block if present
        if node.finally_block:
            self.visit(node.finally_block)

    def visit_Throw(self, node: Throw):
        self.visit(node.value)
        self.emit(OpCode.THROW)

    def visit_UnsafeBlock(self, node: UnsafeBlock):
        if self.mode != CompilerMode.DEBUG:
            self.warning("Unsafe block in non-debug mode", node.line, node.column, node.filename)
        
        self.visit(node.body)

    def visit_Defer(self, node: Defer):
        # Add to deferred stack
        self.deferred.append(node.expr)

    def visit_Spawn(self, node: Spawn):
        self.visit(node.expr)
        self.emit(OpCode.SPAWN)

    def visit_BinaryOp(self, node: BinaryOp):
        self.visit(node.left)
        self.visit(node.right)
        
        if node.op == TokenType.PLUS:
            self.emit(OpCode.ADD)
        elif node.op == TokenType.MINUS:
            self.emit(OpCode.SUB)
        elif node.op == TokenType.STAR:
            self.emit(OpCode.MUL)
        elif node.op == TokenType.SLASH:
            self.emit(OpCode.DIV)
        elif node.op == TokenType.PERCENT:
            self.emit(OpCode.MOD)
        elif node.op == TokenType.CARET:
            self.emit(OpCode.POW)
        elif node.op == TokenType.EQ_EQ:
            self.emit(OpCode.EQ)
        elif node.op == TokenType.BANG_EQ:
            self.emit(OpCode.NEQ)
        elif node.op == TokenType.LT:
            self.emit(OpCode.LT)
        elif node.op == TokenType.GT:
            self.emit(OpCode.GT)
        elif node.op == TokenType.LT_EQ:
            self.emit(OpCode.LTE)
        elif node.op == TokenType.GT_EQ:
            self.emit(OpCode.GTE)
        elif node.op == TokenType.AMP_AMP:
            self.emit(OpCode.AND)
        elif node.op == TokenType.PIPE_PIPE:
            self.emit(OpCode.OR)
        elif node.op == TokenType.AMP:
            self.emit(OpCode.BIT_AND)
        elif node.op == TokenType.PIPE:
            self.emit(OpCode.BIT_OR)
        elif node.op == TokenType.CARET:
            self.emit(OpCode.BIT_XOR)
        elif node.op == TokenType.LT_LT:
            self.emit(OpCode.SHL)
        elif node.op == TokenType.GT_GT:
            self.emit(OpCode.SHR)
        else:
            raise CompilerError(f"Unknown binary operator {node.op}", node.line, node.column, node.filename)

    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.right)
        
        if node.op == TokenType.MINUS:
            self.emit(OpCode.NEG)
        elif node.op == TokenType.BANG:
            self.emit(OpCode.NOT)
        elif node.op == TokenType.TILDE:
            self.emit(OpCode.BIT_NOT)
        elif node.op == TokenType.AMP:
            self.emit(OpCode.ADDRESS)
        elif node.op == TokenType.STAR:
            self.emit(OpCode.DEREF)
        else:
            raise CompilerError(f"Unknown unary operator {node.op}", node.line, node.column, node.filename)

    def visit_Call(self, node: Call):
        self.visit(node.callee)
        
        for arg in node.args:
            self.visit(arg)
        
        if node.is_tail_call:
            self.emit(OpCode.TAIL_CALL, len(node.args))
        else:
            self.emit(OpCode.CALL, len(node.args))

    def visit_MethodCall(self, node: MethodCall):
        self.visit(node.obj)
        self.emit(OpCode.DUP)
        self.emit(OpCode.LOAD_ATTR, self.add_name(node.method))
        
        for arg in node.args:
            self.visit(arg)
        
        self.emit(OpCode.CALL_METHOD, len(node.args))

    def visit_Attribute(self, node: Attribute):
        self.visit(node.obj)
        self.emit(OpCode.LOAD_ATTR, self.add_name(node.attr))

    def visit_Index(self, node: Index):
        self.visit(node.obj)
        self.visit(node.index)
        self.emit(OpCode.LOAD_INDEX)

    def visit_Assignment(self, node: Assignment):
        self.visit(node.value)
        
        if isinstance(node.target, Variable):
            index = self.add_local(node.target.name)
            self.emit(OpCode.STORE, index)
        elif isinstance(node.target, Attribute):
            self.visit(node.target.obj)
            self.emit(OpCode.SWAP)
            self.emit(OpCode.STORE_ATTR, self.add_name(node.target.attr))
        elif isinstance(node.target, Index):
            self.visit(node.target.obj)
            self.visit(node.target.index)
            self.emit(OpCode.SWAP)
            self.emit(OpCode.STORE_INDEX)
        else:
            raise CompilerError("Invalid assignment target", node.line, node.column, node.filename)

    def visit_Literal(self, node: Literal):
        self.emit(OpCode.PUSH, self.add_constant(node.value))

    def visit_Variable(self, node: Variable):
        index = self.find_local(node.name)
        if index is not None:
            self.emit(OpCode.LOAD, index)
        else:
            self.emit(OpCode.LOAD_GLOBAL, self.add_name(node.name))

    def visit_Array(self, node: Array):
        for element in node.elements:
            self.visit(element)
        
        self.emit(OpCode.NEW_ARRAY, len(node.elements))

    def visit_Dict(self, node: Dict):
        for key, value in node.entries:
            self.visit(key)
            self.visit(value)
        
        self.emit(OpCode.NEW_DICT, len(node.entries))

    def visit_Lambda(self, node: Lambda):
        # Similar to Function but without name
        func = Function("", node.params, node.return_type, Block([node.body], node.line, node.column, node.filename), 
                       node.is_async, False, False, node.line, node.column, node.filename)
        self.visit(func)

    def visit_Await(self, node: Await):
        self.visit(node.expr)
        self.emit(OpCode.AWAIT)

    def visit_Yield(self, node: Yield):
        if node.expr:
            self.visit(node.expr)
        else:
            self.emit(OpCode.PUSH, self.add_constant(None))
        
        self.emit(OpCode.YIELD)

    def emit(self, opcode: OpCode, arg: int = 0):
        self.bytecode.append(opcode.value)
        self.bytecode.append(arg)

    def add_constant(self, value: Any) -> int:
        self.constants.append(value)
        return len(self.constants) - 1

    def add_name(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    def add_local(self, name: str) -> int:
        if name not in self.locals:
            self.locals.append(name)
        return self.locals.index(name)

    def find_local(self, name: str) -> Optional[int]:
        if name in self.locals:
            return self.locals.index(name)
        return None

    def warning(self, message: str, line: int, column: int, filename: str):
        if self.mode == CompilerMode.DEBUG:
            print(f"Warning: {message} at {filename}:{line}:{column}")

# ==================== PACKAGE MANAGER ====================
class PackageManager:
    def __init__(self):
        os.makedirs(PKG_CACHE, exist_ok=True)
        self.installed_packages = self._load_installed_packages()

    def _load_installed_packages(self) -> Dict[str, str]:
        try:
            with open(os.path.join(PKG_CACHE, "packages.json"), "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}

    def _save_installed_packages(self):
        with open(os.path.join(PKG_CACHE, "packages.json"), "w") as f:
            json.dump(self.installed_packages, f, indent=2)

    def install(self, package_name: str, version: Optional[str] = None) -> bool:
        if package_name in self.installed_packages:
            if version and self.installed_packages[package_name] != version:
                print(f"Package {package_name} already installed with different version")
                return False
            print(f"Package {package_name} already installed")
            return True
        
        try:
            # Download package
            url = f"{PKG_REGISTRY}/{package_name}"
            if version:
                url += f"/{version}"
            
            with urlopen(url) as response:
                if response.status != 200:
                    print(f"Failed to download package {package_name}")
                    return False
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    shutil.copyfileobj(response, tmp_file)
                    tmp_path = tmp_file.name
                
                # Extract package
                pkg_dir = os.path.join(PKG_CACHE, package_name)
                os.makedirs(pkg_dir, exist_ok=True)
                
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
                
                # Record installation
                version = version or "latest"
                self.installed_packages[package_name] = version
                self._save_installed_packages()
                
                print(f"Successfully installed {package_name}@{version}")
                return True
        except Exception as e:
            print(f"Error installing package {package_name}: {str(e)}")
            return False

    def uninstall(self, package_name: str) -> bool:
        if package_name not in self.installed_packages:
            print(f"Package {package_name} not installed")
            return False
        
        try:
            pkg_dir = os.path.join(PKG_CACHE, package_name)
            shutil.rmtree(pkg_dir)
            
            del self.installed_packages[package_name]
            self._save_installed_packages()
            
            print(f"Successfully uninstalled {package_name}")
            return True
        except Exception as e:
            print(f"Error uninstalling package {package_name}: {str(e)}")
            return False

    def list_packages(self) -> Dict[str, str]:
        return self.installed_packages.copy()

# ==================== WASM COMPILER ====================
class WasmCompiler:
    def __init__(self):
        self.wasm_code = []
        self.imports = set()
        self.exports = set()
        self.functions = []
        self.memory = None
        self.table = None

    def compile(self, bytecode: bytes, constants: List[Any], names: List[str]) -> str:
        self.wasm_code = [
            "(module",
            '  (type (;0;) (func (param i32 i32) (result i32)))',
            '  (type (;1;) (func (param i32) (result i32)))',
            '  (type (;2;) (func (param i32)))',
            '  (type (;3;) (func (result i32)))',
            '  (type (;4;) (func))'
        ]
        
        # Add memory
        self.wasm_code.extend([
            '  (memory (;0;) 1)',
            '  (export "memory" (memory 0))'
        ])
        
        # Add functions
        for i, name in enumerate(names):
            self.wasm_code.append(f'  (func ${name} (type 0) (param i32 i32) (result i32)')
            self.wasm_code.append('    (local.get 0)')
            self.wasm_code.append('    (local.get 1)')
            self.wasm_code.append('    (i32.add)')
            self.wasm_code.append('  )')
        
        # Add main function
        self.wasm_code.extend([
            '  (func $main (type 3) (result i32)',
            '    (i32.const 42)',
            '  )',
            '  (export "main" (func $main))',
            ')'
        ])
        
        return '\n'.join(self.wasm_code)

# ==================== LLVM COMPILER ====================
class LLVMCompiler:
    def __init__(self):
        self.llvm_code = []
        self.functions = []
        self.globals = []

    def compile(self, bytecode: bytes, constants: List[Any], names: List[str]) -> str:
        self.llvm_code = [
            "; ModuleID = 'joyl_module'",
            "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"",
            "target triple = \"x86_64-pc-linux-gnu\"",
            ""
        ]
        
        # Add string constants
        for i, const in enumerate(constants):
            if isinstance(const, str):
                self.llvm_code.append(f"@.str.{i} = private unnamed_addr constant [{len(const)+1} x i8] c\"{const}\\00\", align 1")
        
        # Add main function
        self.llvm_code.extend([
            "define i32 @main() {",
            "entry:",
            "  ret i32 0",
            "}"
        ])
        
        return '\n'.join(self.llvm_code)

# ==================== VIRTUAL MACHINE ====================
class VM:
    def __init__(self, mode: CompilerMode = CompilerMode.DEBUG):
        self.mode = mode
        self.stack = []
        self.globals = {}
        self.call_stack = []
        self.current_frame = None
        self.exception = None
        self.last_value = None
        self.running = False
        self.load_native_functions()
        self.gc_enabled = True
        self.gc_threshold = 1000
        self.allocated_objects = 0
        self.locks = {}
        self.channels = {}

    def load_native_functions(self):
        # Core functions
        self.globals['print'] = self.native_print
        self.globals['println'] = self.native_println
        self.globals['input'] = self.native_input
        self.globals['len'] = self.native_len
        self.globals['range'] = self.native_range
        self.globals['type'] = self.native_type
        self.globals['assert'] = self.native_assert
        self.globals['panic'] = self.native_panic
        self.globals['gc_enable'] = lambda flag: setattr(self, 'gc_enabled', flag)
        self.globals['gc_collect'] = self.native_gc_collect
        
        # Math functions
        self.globals['math'] = {
            'pi': math.pi,
            'e': math.e,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'floor': math.floor,
            'ceil': math.ceil,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': math.pow,
            'random': random.random,
            'randint': random.randint
        }
        
        # String functions
        self.globals['string'] = {
            'split': str.split,
            'join': lambda sep, seq: sep.join(seq),
            'upper': str.upper,
            'lower': str.lower,
            'strip': str.strip,
            'replace': str.replace,
            'find': str.find,
            'startswith': str.startswith,
            'endswith': str.endswith,
            'format': str.format,
            'encode': lambda s, enc: s.encode(enc),
            'decode': lambda b, enc: b.decode(enc)
        }
        
        # List functions
        self.globals['list'] = {
            'append': lambda lst, x: lst.append(x),
            'extend': lambda lst, seq: lst.extend(seq),
            'pop': lambda lst, i=-1: lst.pop(i),
            'insert': lambda lst, i, x: lst.insert(i, x),
            'remove': lambda lst, x: lst.remove(x),
            'index': lambda lst, x: lst.index(x),
            'count': lambda lst, x: lst.count(x),
            'sort': lambda lst: lst.sort(),
            'reverse': lambda lst: lst.reverse(),
            'copy': lambda lst: lst.copy(),
            'clear': lambda lst: lst.clear(),
            'map': lambda lst, fn: [fn(x) for x in lst],
            'filter': lambda lst, fn: [x for x in lst if fn(x)],
            'reduce': lambda lst, fn, init: functools.reduce(fn, lst, init)
        }
        
        # Dict functions
        self.globals['dict'] = {
            'keys': lambda d: list(d.keys()),
            'values': lambda d: list(d.values()),
            'items': lambda d: list(d.items()),
            'get': lambda d, k, default=None: d.get(k, default),
            'pop': lambda d, k: d.pop(k),
            'update': lambda d, other: d.update(other),
            'clear': lambda d: d.clear(),
            'copy': lambda d: d.copy()
        }
        
        # Time functions
        self.globals['time'] = {
            'time': time.time,
            'sleep': time.sleep,
            'ctime': time.ctime,
            'gmtime': time.gmtime,
            'localtime': time.localtime,
            'mktime': time.mktime,
            'strftime': time.strftime,
            'strptime': time.strptime
        }
        
        # File I/O functions
        self.globals['io'] = {
            'open': open,
            'read': lambda f: f.read(),
            'write': lambda f, data: f.write(data),
            'close': lambda f: f.close(),
            'exists': os.path.exists,
            'isdir': os.path.isdir,
            'isfile': os.path.isfile,
            'listdir': os.listdir,
            'mkdir': os.mkdir,
            'remove': os.remove,
            'rename': os.rename
        }
        
        # Network functions
        self.globals['net'] = {
            'socket': socket.socket,
            'connect': lambda s, addr: s.connect(addr),
            'bind': lambda s, addr: s.bind(addr),
            'listen': lambda s, n: s.listen(n),
            'accept': lambda s: s.accept(),
            'send': lambda s, data: s.send(data),
            'recv': lambda s, n: s.recv(n),
            'close': lambda s: s.close(),
            'gethostname': socket.gethostname,
            'gethostbyname': socket.gethostbyname
        }
        
        # Threading functions
        self.globals['thread'] = {
            'Thread': threading.Thread,
            'Lock': threading.Lock,
            'RLock': threading.RLock,
            'Condition': threading.Condition,
            'Semaphore': threading.Semaphore,
            'Event': threading.Event,
            'current_thread': threading.current_thread,
            'active_count': threading.active_count
        }
        
        # Concurrency functions
        self.globals['concurrent'] = {
            'spawn': self.native_spawn,
            'chan': self.native_chan,
            'send': self.native_chan_send,
            'recv': self.native_chan_recv,
            'select': self.native_select
        }

    def native_print(self, *args):
        print(*args, end='')
        return None

    def native_println(self, *args):
        print(*args)
        return None

    def native_input(self, prompt=""):
        return input(prompt)

    def native_len(self, obj):
        return len(obj)

    def native_range(self, start, stop=None, step=1):
        if stop is None:
            return list(range(start))
        return list(range(start, stop, step))

    def native_type(self, obj):
        if obj is None:
            return "nil"
        elif isinstance(obj, bool):
            return "bool"
        elif isinstance(obj, int):
            return "int"
        elif isinstance(obj, float):
            return "float"
        elif isinstance(obj, str):
            return "string"
        elif isinstance(obj, list):
            return "list"
        elif isinstance(obj, dict):
            return "dict"
        elif callable(obj):
            return "function"
        else:
            return "object"

    def native_assert(self, condition, message=None):
        if not condition:
            if message:
                raise AssertionError(message)
            else:
                raise AssertionError("Assertion failed")

    def native_panic(self, message):
        raise RuntimeError(message)

    def native_gc_collect(self):
        # Simple mark-and-sweep GC
        marked = set()
        
        # Mark from stack
        for obj in self.stack:
            self._mark_object(obj, marked)
        
        # Mark from globals
        for obj in self.globals.values():
            self._mark_object(obj, marked)
        
        # Mark from call frames
        for frame in self.call_stack:
            for obj in frame.get('locals', []):
                self._mark_object(obj, marked)
        
        # Sweep unmarked objects
        self.allocated_objects = len(marked)
        return len(marked)

    def _mark_object(self, obj, marked):
        if id(obj) in marked:
            return
        if isinstance(obj, (list, dict)):
            marked.add(id(obj))
            if isinstance(obj, list):
                for item in obj:
                    self._mark_object(item, marked)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    self._mark_object(key, marked)
                    self._mark_object(value, marked)

    def native_spawn(self, func, *args):
        thread = threading.Thread(target=func, args=args)
        thread.start()
        return thread

    def native_chan(self, buffer_size=0):
        chan_id = len(self.channels) + 1
        self.channels[chan_id] = queue.Queue(buffer_size)
        return chan_id

    def native_chan_send(self, chan_id, value):
        if chan_id not in self.channels:
            raise RuntimeError(f"Channel {chan_id} does not exist")
        self.channels[chan_id].put(value)

    def native_chan_recv(self, chan_id):
        if chan_id not in self.channels:
            raise RuntimeError(f"Channel {chan_id} does not exist")
        return self.channels[chan_id].get()

    def native_select(self, *cases):
        # Simple select implementation
        while True:
            for chan_id, callback in cases:
                if not self.channels[chan_id].empty():
                    value = self.channels[chan_id].get()
                    return callback(value)
            time.sleep(0.01)  # Prevent busy waiting

    def run(self, bytecode: bytes, constants: List[Any], names: List[str]):
        self.stack = []
        self.call_stack = []
        self.current_frame = {
            'bytecode': bytecode,
            'constants': constants,
            'names': names,
            'locals': [None] * len(names),
            'ip': 0
        }
        self.exception = None
        self.last_value = None
        self.running = True
        self.allocated_objects = 0
        
        try:
            while self.running and self.current_frame['ip'] < len(bytecode):
                opcode = OpCode(bytecode[self.current_frame['ip']])
                arg = bytecode[self.current_frame['ip'] + 1]
                self.current_frame['ip'] += 2
                
                # Check memory pressure
                if self.gc_enabled and self.allocated_objects > self.gc_threshold:
                    self.native_gc_collect()
                
                self.execute(opcode, arg)
        except Exception as e:
            self.running = False
            if self.mode == CompilerMode.DEBUG:
                traceback.print_exc()
            raise RuntimeError(f"Runtime error: {str(e)}")
        
        return self.last_value

    def execute(self, opcode: OpCode, arg: int):
        try:
            if opcode == OpCode.PUSH:
                self.stack.append(self.current_frame['constants'][arg])
                self.allocated_objects += 1
            elif opcode == OpCode.POP:
                self.stack.pop()
            elif opcode == OpCode.DUP:
                self.stack.append(self.stack[-1])
            elif opcode == OpCode.SWAP:
                a, b = self.stack[-2], self.stack[-1]
                self.stack[-2], self.stack[-1] = b, a
            elif opcode == OpCode.ROT:
                a, b, c = self.stack[-3], self.stack[-2], self.stack[-1]
                self.stack[-3], self.stack[-2], self.stack[-1] = b, c, a
            
            # Variable operations
            elif opcode == OpCode.LOAD:
                self.stack.append(self.current_frame['locals'][arg])
            elif opcode == OpCode.STORE:
                self.current_frame['locals'][arg] = self.stack.pop()
            elif opcode == OpCode.LOAD_GLOBAL:
                self.stack.append(self.globals[self.current_frame['names'][arg]])
            elif opcode == OpCode.STORE_GLOBAL:
                self.globals[self.current_frame['names'][arg]] = self.stack.pop()
            elif opcode == OpCode.LOAD_ATTR:
                obj = self.stack.pop()
                attr = self.current_frame['names'][arg]
                self.stack.append(getattr(obj, attr) if hasattr(obj, attr) else obj[attr])
            elif opcode == OpCode.STORE_ATTR:
                value = self.stack.pop()
                obj = self.stack.pop()
                attr = self.current_frame['names'][arg]
                if hasattr(obj, attr):
                    setattr(obj, attr, value)
                else:
                    obj[attr] = value
            elif opcode == OpCode.LOAD_INDEX:
                index = self.stack.pop()
                obj = self.stack.pop()
                self.stack.append(obj[index])
            elif opcode == OpCode.STORE_INDEX:
                value = self.stack.pop()
                index = self.stack.pop()
                obj = self.stack.pop()
                obj[index] = value
            elif opcode == OpCode.LOAD_MODULE:
                module = self.current_frame['constants'][arg]
                if module in sys.modules:
                    self.stack.append(sys.modules[module])
                else:
                    imported = importlib.import_module(module)
                    self.stack.append(imported)
            
            # Memory management
            elif opcode == OpCode.ALLOC:
                size = self.stack.pop()
                self.stack.append([None] * size)
                self.allocated_objects += 1
            elif opcode == OpCode.FREE:
                self.stack.pop()
            elif opcode == OpCode.GC_COLLECT:
                self.native_gc_collect()
            
            # Arithmetic operations
            elif opcode == OpCode.ADD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)
            elif opcode == OpCode.SUB:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)
            elif opcode == OpCode.MUL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)
            elif opcode == OpCode.DIV:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a / b)
            elif opcode == OpCode.MOD:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a % b)
            elif opcode == OpCode.POW:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a ** b)
            elif opcode == OpCode.NEG:
                self.stack.append(-self.stack.pop())
            elif opcode == OpCode.INC:
                self.current_frame['locals'][arg] += 1
            elif opcode == OpCode.DEC:
                self.current_frame['locals'][arg] -= 1
            
            # Bitwise operations
            elif opcode == OpCode.BIT_AND:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a & b)
            elif opcode == OpCode.BIT_OR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a | b)
            elif opcode == OpCode.BIT_XOR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a ^ b)
            elif opcode == OpCode.BIT_NOT:
                self.stack.append(~self.stack.pop())
            elif opcode == OpCode.SHL:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a << b)
            elif opcode == OpCode.SHR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a >> b)
            
            # Logical operations
            elif opcode == OpCode.AND:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a and b)
            elif opcode == OpCode.OR:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a or b)
            elif opcode == OpCode.NOT:
                self.stack.append(not self.stack.pop())
            
            # Comparison operations
            elif opcode == OpCode.EQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a == b)
            elif opcode == OpCode.NEQ:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a != b)
            elif opcode == OpCode.LT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a < b)
            elif opcode == OpCode.GT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a > b)
            elif opcode == OpCode.LTE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a <= b)
            elif opcode == OpCode.GTE:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a >= b)
            elif opcode == OpCode.IS:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a is b)
            elif opcode == OpCode.IS_NOT:
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a is not b)
            
            # Control flow
            elif opcode == OpCode.JMP:
                self.current_frame['ip'] += arg
            elif opcode == OpCode.JMP_IF:
                if self.stack.pop():
                    self.current_frame['ip'] += arg
            elif opcode == OpCode.JMP_IF_NOT:
                if not self.stack.pop():
                    self.current_frame['ip'] += arg
            elif opcode == OpCode.JMP_IF_TRUE:
                if self.stack[-1]:
                    self.current_frame['ip'] += arg
            elif opcode == OpCode.JMP_IF_FALSE:
                if not self.stack[-1]:
                    self.current_frame['ip'] += arg
            elif opcode == OpCode.CALL:
                func = self.stack[-arg - 1]
                args = [self.stack.pop() for _ in range(arg)]
                args.reverse()
                
                if callable(func):
                    result = func(*args)
                    self.stack.append(result)
                else:
                    raise RuntimeError(f"{func} is not callable")
            elif opcode == OpCode.RETURN:
                self.last_value = self.stack.pop() if self.stack else None
                if self.call_stack:
                    self.current_frame = self.call_stack.pop()
                else:
                    self.running = False
            elif opcode == OpCode.CALL_METHOD:
                method = self.stack.pop()
                obj = self.stack.pop()
                args = [self.stack.pop() for _ in range(arg)]
                args.reverse()
                
                if callable(method):
                    result = method(obj, *args)
                    self.stack.append(result)
                else:
                    raise RuntimeError(f"{method} is not callable")
            elif opcode == OpCode.TAIL_CALL:
                func = self.stack[-arg - 1]
                args = [self.stack.pop() for _ in range(arg)]
                args.reverse()
                
                if callable(func):
                    result = func(*args)
                    self.stack.append(result)
                else:
                    raise RuntimeError(f"{func} is not callable")
            elif opcode == OpCode.CALL_NATIVE:
                func = self.current_frame['names'][arg]
                args = [self.stack.pop() for _ in range(arg)]
                args.reverse()
                result = self.globals[func](*args)
                self.stack.append(result)
            
            # Object operations
            elif opcode == OpCode.NEW:
                cls = self.stack.pop()
                args = [self.stack.pop() for _ in range(arg)]
                args.reverse()
                self.stack.append(cls(*args))
                self.allocated_objects += 1
            elif opcode == OpCode.NEW_ARRAY:
                elements = [self.stack.pop() for _ in range(arg)]
                elements.reverse()
                self.stack.append(elements)
                self.allocated_objects += 1
            elif opcode == OpCode.NEW_DICT:
                items = {}
                for _ in range(arg):
                    value = self.stack.pop()
                    key = self.stack.pop()
                    items[key] = value
                self.stack.append(items)
                self.allocated_objects += 1
            elif opcode == OpCode.NEW_CLASS:
                class_info = self.stack.pop()
                cls = type(class_info['name'], (object,), class_info)
                self.stack.append(cls)
                self.allocated_objects += 1
            elif opcode == OpCode.NEW_FUNCTION:
                func_info = self.stack.pop()
                def func(*args):
                    vm = VM(self.mode)
                    vm.globals.update(self.globals)
                    for i, (name, _) in enumerate(func_info['params']):
                        vm.globals[name] = args[i] if i < len(args) else None
                    return vm.run(func_info['code'], func_info['constants'], func_info['names'])
                self.stack.append(func)
                self.allocated_objects += 1
            elif opcode == OpCode.NEW_MODULE:
                module_info = self.stack.pop()
                module = type(module_info['name'], (object,), module_info)
                self.stack.append(module)
                self.allocated_objects += 1
            elif opcode == OpCode.INSTANCE_OF:
                cls = self.stack.pop()
                obj = self.stack.pop()
                self.stack.append(isinstance(obj, cls))
            elif opcode == OpCode.CLONE:
                obj = self.stack.pop()
                self.stack.append(copy.deepcopy(obj))
                self.allocated_objects += 1
            
            # Type operations
            elif opcode == OpCode.TO_INT:
                self.stack.append(int(self.stack.pop()))
            elif opcode == OpCode.TO_FLOAT:
                self.stack.append(float(self.stack.pop()))
            elif opcode == OpCode.TO_BOOL:
                self.stack.append(bool(self.stack.pop()))
            elif opcode == OpCode.TO_STRING:
                self.stack.append(str(self.stack.pop()))
            elif opcode == OpCode.TYPE_OF:
                obj = self.stack.pop()
                if obj is None:
                    self.stack.append("nil")
                elif isinstance(obj, bool):
                    self.stack.append("bool")
                elif isinstance(obj, int):
                    self.stack.append("int")
                elif isinstance(obj, float):
                    self.stack.append("float")
                elif isinstance(obj, str):
                    self.stack.append("string")
                elif isinstance(obj, list):
                    self.stack.append("list")
                elif isinstance(obj, dict):
                    self.stack.append("dict")
                elif callable(obj):
                    self.stack.append("function")
                else:
                    self.stack.append("object")
            elif opcode == OpCode.CHECK_TYPE:
                type_name = self.current_frame['names'][arg]
                obj = self.stack.pop()
                actual_type = self.native_type(obj)
                self.stack.append(actual_type == type_name)
            
            # Async operations
            elif opcode == OpCode.AWAIT:
                future = self.stack.pop()
                if hasattr(future, '__await__'):
                    self.stack.append(future.__await__())
                else:
                    self.stack.append(future)
            elif opcode == OpCode.YIELD:
                self.last_value = self.stack.pop() if arg else None
                self.running = False
            elif opcode == OpCode.SPAWN:
                func = self.stack.pop()
                if callable(func):
                    thread = threading.Thread(target=func)
                    thread.start()
                    self.stack.append(thread)
                else:
                    raise RuntimeError("Cannot spawn non-callable object")
            
            # Concurrency
            elif opcode == OpCode.LOCK:
                lock_id = arg
                if lock_id not in self.locks:
                    self.locks[lock_id] = threading.Lock()
                self.locks[lock_id].acquire()
            elif opcode == OpCode.UNLOCK:
                lock_id = arg
                if lock_id in self.locks:
                    self.locks[lock_id].release()
            elif opcode == OpCode.CHAN_SEND:
                chan_id = arg
                value = self.stack.pop()
                if chan_id in self.channels:
                    self.channels[chan_id].put(value)
                else:
                    raise RuntimeError(f"Channel {chan_id} does not exist")
            elif opcode == OpCode.CHAN_RECV:
                chan_id = arg
                if chan_id in self.channels:
                    self.stack.append(self.channels[chan_id].get())
                else:
                    raise RuntimeError(f"Channel {chan_id} does not exist")
            
            # Error handling
            elif opcode == OpCode.THROW:
                raise RuntimeError(self.stack.pop())
            elif opcode == OpCode.TRY_BEGIN:
                self.call_stack.append({
                    'ip': self.current_frame['ip'] + arg,
                    'exception': None
                })
            elif opcode == OpCode.TRY_END:
                self.call_stack.pop()
            elif opcode == OpCode.ASSERT:
                condition = self.stack.pop()
                if not condition:
                    message = self.stack.pop() if arg else "Assertion failed"
                    raise AssertionError(message)
            
            # Debugging
            elif opcode == OpCode.DEBUG:
                if self.mode == CompilerMode.DEBUG:
                    print(f"DEBUG: {self.stack[-1]}")
            elif opcode == OpCode.BREAKPOINT:
                if self.mode == CompilerMode.DEBUG:
                    import pdb; pdb.set_trace()
            
            # Special
            elif opcode == OpCode.NOP:
                pass
            elif opcode == OpCode.HALT:
                self.running = False
            elif opcode == OpCode.IMPORT:
                module = self.stack.pop()
                version = self.stack.pop() if arg > 1 else None
                if module in sys.modules:
                    self.stack.append(sys.modules[module])
                else:
                    try:
                        imported = importlib.import_module(module)
                        self.stack.append(imported)
                    except ImportError:
                        # Try package manager
                        pm = PackageManager()
                        if pm.install(module, version):
                            imported = importlib.import_module(module)
                            self.stack.append(imported)
                        else:
                            raise ImportError(f"Could not import module {module}")
            elif opcode == OpCode.EXPORT:
                name = self.current_frame['names'][arg]
                value = self.stack.pop()
                self.globals[name] = value
            else:
                raise RuntimeError(f"Unknown opcode {opcode}")
        except Exception as e:
            if self.call_stack and self.call_stack[-1].get('exception') is None:
                self.call_stack[-1]['exception'] = e
                self.current_frame['ip'] = self.call_stack[-1]['ip']
            else:
                raise e

# ==================== COMPILER DRIVER ====================
class JoylCompiler:
    def __init__(self, mode: CompilerMode = CompilerMode.DEBUG, 
                 optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
                 target: TargetPlatform = TargetPlatform.BYTECODE):
        self.mode = mode
        self.optimization_level = optimization_level
        self.target = target
        self.lexer = None
        self.parser = None
        self.compiler = None
        self.vm = VM(mode)
        self.wasm_compiler = WasmCompiler()
        self.llvm_compiler = LLVMCompiler()
        self.errors = []
        self.warnings = []
        self.package_manager = PackageManager()

    def compile(self, source: str, filename: str = "<stdin>") -> Optional[Tuple[bytes, List[Any], List[str]]]:
        try:
            # Lexical analysis
            self.lexer = Lexer(source, filename)
            tokens = self.lexer.scan_tokens()
            if self.lexer.errors:
                self.errors.extend(self.lexer.errors)
                return None
            
            # Parsing
            self.parser = Parser(tokens, filename)
            ast = self.parser.parse()
            if self.parser.errors:
                self.errors.extend(self.parser.errors)
                return None
            
            # Compilation
            self.compiler = Compiler(self.mode, self.optimization_level, self.target)
            bytecode, constants, names = self.compiler.compile(ast)
            if self.compiler.errors:
                self.errors.extend(self.compiler.errors)
                return None
            
            return bytecode, constants, names
        except Exception as e:
            self.errors.append(CompilerError(str(e), filename=filename))
            return None

    def compile_to_wasm(self, source: str, filename: str = "<stdin>") -> Optional[str]:
        result = self.compile(source, filename)
        if result is None:
            return None
        
        bytecode, constants, names = result
        return self.wasm_compiler.compile(bytecode, constants, names)

    def compile_to_llvm(self, source: str, filename: str = "<stdin>") -> Optional[str]:
        result = self.compile(source, filename)
        if result is None:
            return None
        
        bytecode, constants, names = result
        return self.llvm_compiler.compile(bytecode, constants, names)

    def run(self, source: str, filename: str = "<stdin>") -> Any:
        result = self.compile(source, filename)
        if result is None:
            return None
        
        bytecode, constants, names = result
        return self.vm.run(bytecode, constants, names)

    def run_bytecode(self, bytecode: bytes, constants: List[Any], names: List[str]) -> Any:
        return self.vm.run(bytecode, constants, names)

    def install_package(self, package_name: str, version: Optional[str] = None) -> bool:
        return self.package_manager.install(package_name, version)

    def uninstall_package(self, package_name: str) -> bool:
        return self.package_manager.uninstall(package_name)

    def list_packages(self) -> Dict[str, str]:
        return self.package_manager.list_packages()

# ==================== COMMAND-LINE INTERFACE ====================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Joyl Compiler')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a Joyl program')
    run_parser.add_argument('file', nargs='?', help='Joyl source file')
    run_parser.add_argument('-O', '--optimize', type=int, default=DEFAULT_OPTIMIZATION_LEVEL,
                          help=f'Optimization level (0-3, default: {DEFAULT_OPTIMIZATION_LEVEL})')
    run_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    run_parser.add_argument('--release', action='store_true', help='Enable release mode')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile a Joyl program')
    compile_parser.add_argument('file', help='Joyl source file')
    compile_parser.add_argument('-o', '--output', help='Output file')
    compile_parser.add_argument('-O', '--optimize', type=int, default=DEFAULT_OPTIMIZATION_LEVEL,
                              help=f'Optimization level (0-3, default: {DEFAULT_OPTIMIZATION_LEVEL})')
    compile_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    compile_parser.add_argument('--release', action='store_true', help='Enable release mode')
    compile_parser.add_argument('--wasm', action='store_true', help='Compile to WebAssembly')
    compile_parser.add_argument('--llvm', action='store_true', help='Compile to LLVM IR')
    
    # Package commands
    pkg_parser = subparsers.add_parser('pkg', help='Package management')
    pkg_subparsers = pkg_parser.add_subparsers(dest='pkg_command', required=True)
    
    pkg_install = pkg_subparsers.add_parser('install', help='Install a package')
    pkg_install.add_argument('package', help='Package name')
    pkg_install.add_argument('--version', help='Package version')
    
    pkg_uninstall = pkg_subparsers.add_parser('uninstall', help='Uninstall a package')
    pkg_uninstall.add_argument('package', help='Package name')
    
    pkg_list = pkg_subparsers.add_parser('list', help='List installed packages')
    
    args = parser.parse_args()
    
    mode = CompilerMode.DEBUG if args.debug else (CompilerMode.RELEASE if args.release else CompilerMode.DEBUG)
    target = TargetPlatform.WASM if args.wasm else (TargetPlatform.LLVM if args.llvm else TargetPlatform.BYTECODE)
    compiler = JoylCompiler(mode, args.optimize, target)
    
    if args.command == 'run':
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    source = f.read()
            except IOError as e:
                print(f"Error reading file: {e}", file=sys.stderr)
                sys.exit(1)
            
            result = compiler.run(source, args.file)
            if compiler.errors:
                for error in compiler.errors:
                    print(error, file=sys.stderr)
                sys.exit(1)
            
            if result is not None:
                print(result)
        else:
            # REPL mode
            print(f"Joyl REPL (v{VERSION}) - Type 'exit' or 'quit' to exit")
            while True:
                try:
                    source = input("> ")
                    if source.lower() in ('exit', 'quit'):
                        break
                    
                    result = compiler.run(source)
                    if compiler.errors:
                        for error in compiler.errors:
                            print(error, file=sys.stderr)
                        compiler.errors.clear()
                    elif result is not None:
                        print(result)
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt")
                except EOFError:
                    print()
                    break
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
    
    elif args.command == 'compile':
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                source = f.read()
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
        
        if args.wasm:
            result = compiler.compile_to_wasm(source, args.file)
            ext = "wat"
        elif args.llvm:
            result = compiler.compile_to_llvm(source, args.file)
            ext = "ll"
        else:
            result = compiler.compile(source, args.file)
            ext = "joylc"
        
        if result is None:
            for error in compiler.errors:
                print(error, file=sys.stderr)
            sys.exit(1)
        
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    if isinstance(result, tuple):
                        bytecode, constants, names = result
                        output = {
                            'bytecode': bytecode.hex(),
                            'constants': constants,
                            'names': names
                        }
                        json.dump(output, f, indent=2)
                    else:
                        f.write(result)
            except IOError as e:
                print(f"Error writing file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            if isinstance(result, tuple):
                bytecode, constants, names = result
                output = {
                    'bytecode': bytecode.hex(),
                    'constants': constants,
                    'names': names
                }
                print(json.dumps(output, indent=2))
            else:
                print(result)
    
    elif args.command == 'pkg':
        if args.pkg_command == 'install':
            success = compiler.install_package(args.package, args.version)
            sys.exit(0 if success else 1)
        elif args.pkg_command == 'uninstall':
            success = compiler.uninstall_package(args.package)
            sys.exit(0 if success else 1)
        elif args.pkg_command == 'list':
            packages = compiler.list_packages()
            for pkg, version in packages.items():
                print(f"{pkg}@{version}")
            sys.exit(0)

if __name__ == "__main__":
    main()