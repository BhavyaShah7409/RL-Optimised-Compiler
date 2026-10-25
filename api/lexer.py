from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"

    # Keywords
    PRINT = "PRINT"
    SCAN = "SCAN"
    WHILE = "WHILE"
    IF = "IF"
    ELSE = "ELSE"

    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    ASSIGN = "ASSIGN"
    PLUS_ASSIGN = "PLUS_ASSIGN"
    MINUS_ASSIGN = "MINUS_ASSIGN"

    # Comparison
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    GREATER = "GREATER"
    LESS = "LESS"
    GREATER_EQUAL = "GREATER_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"

    # Logical
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"   # BUG FIX #1: was "RBRACE" — caused RPAREN == RBRACE
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    SEMICOLON = "SEMICOLON"

    # Special
    NEWLINE = "NEWLINE"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class Lexer:
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []

        self.keywords = {
            'print': TokenType.PRINT,
            'scan': TokenType.SCAN,
            'while': TokenType.WHILE,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
        }

        # Two-char operators must be checked before single-char ones
        self.two_char_operators = {
            '+=': TokenType.PLUS_ASSIGN,
            '-=': TokenType.MINUS_ASSIGN,
            '==': TokenType.EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '&&': TokenType.AND,
            '||': TokenType.OR,
        }

        self.single_char_operators = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '=': TokenType.ASSIGN,
            '>': TokenType.GREATER,
            '<': TokenType.LESS,
            '!': TokenType.NOT,
        }

        self.delimiters = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            ';': TokenType.SEMICOLON,
        }

    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]

    def advance(self):
        if self.position < len(self.source) and self.source[self.position] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.position += 1

    def skip_whitespace(self):
        # BUG FIX #2: skip spaces, tabs, CR — but NOT newlines (handled separately)
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()

    def read_number(self) -> str:
        start_pos = self.position
        dot_seen = False
        while self.current_char() and (self.current_char().isdigit() or
                                        (self.current_char() == '.' and not dot_seen)):
            # BUG FIX #14: only allow one decimal point
            if self.current_char() == '.':
                dot_seen = True
            self.advance()
        return self.source[start_pos:self.position]

    def read_string(self, quote_char: str) -> str:
        """Read a string literal delimited by quote_char (single or double)."""
        self.advance()  # Skip opening quote
        start_pos = self.position
        result_chars = []
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\n':
                raise SyntaxError(f"Unterminated string literal at line {self.line}")
            if self.current_char() == '\\':
                self.advance()  # skip backslash
                escape_map = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'"}
                c = self.current_char()
                result_chars.append(escape_map.get(c, c) if c else '')
            else:
                result_chars.append(self.current_char())
            self.advance()

        if not self.current_char():
            raise SyntaxError(f"Unterminated string at line {self.line}")

        value = ''.join(result_chars)
        self.advance()  # Skip closing quote
        return value

    def read_identifier(self) -> str:
        start_pos = self.position
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            self.advance()
        return self.source[start_pos:self.position]

    def tokenize(self) -> List[Token]:
        while self.position < len(self.source):
            self.skip_whitespace()

            if not self.current_char():
                break

            char = self.current_char()
            line, column = self.line, self.column

            # Handle single-line comments
            if char == '/' and self.peek_char() == '/':
                while self.current_char() and self.current_char() != '\n':
                    self.advance()
                continue

            # BUG FIX #2: Newlines are only handled here — not duplicated in the
            # operator/delimiter branch below
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, char, line, column))
                self.advance()
                continue

            # Numbers
            if char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, line, column))
                continue

            # Strings — BUG FIX #13: support both single and double quote strings
            if char in ('"', "'"):
                value = self.read_string(char)
                self.tokens.append(Token(TokenType.STRING, value, line, column))
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                value = self.read_identifier()
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, line, column))
                continue

            # Delimiters
            if char in self.delimiters:
                self.tokens.append(Token(self.delimiters[char], char, line, column))
                self.advance()
                continue

            # Two-character operators (must be checked before single-char)
            next_char = self.peek_char()
            two_char = char + (next_char or '')
            if two_char in self.two_char_operators:
                self.tokens.append(Token(self.two_char_operators[two_char], two_char, line, column))
                self.advance()
                self.advance()
                continue

            # Single-character operators
            if char in self.single_char_operators:
                self.tokens.append(Token(self.single_char_operators[char], char, line, column))
                self.advance()
                continue

            raise SyntaxError(f"Unknown character '{char}' at line {line}, column {column}")

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
