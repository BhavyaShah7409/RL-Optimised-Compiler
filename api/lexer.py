import re
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
    RPAREN = "RBRACE"
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
        
        self.operators = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '=': TokenType.ASSIGN,
            '+=': TokenType.PLUS_ASSIGN,
            '-=': TokenType.MINUS_ASSIGN,
            '==': TokenType.EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '>': TokenType.GREATER,
            '<': TokenType.LESS,
            '>=': TokenType.GREATER_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '&&': TokenType.AND,
            '||': TokenType.OR,
            '!': TokenType.NOT,
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
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def read_number(self) -> str:
        start_pos = self.position
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            self.advance()
        return self.source[start_pos:self.position]
    
    def read_string(self) -> str:
        self.advance()  # Skip opening quote
        start_pos = self.position
        while self.current_char() and self.current_char() != '"':
            if self.current_char() == '\\':
                self.advance()  # Skip escape character
            self.advance()
        
        if not self.current_char():
            raise SyntaxError(f"Unterminated string at line {self.line}")
        
        value = self.source[start_pos:self.position]
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
            
            # Handle comments
            char = self.current_char()
            if char == '/' and self.peek_char() == '/':
                # Skip single-line comment
                while self.current_char() and self.current_char() != '\n':
                    self.advance()
                continue
            
            line, column = self.line, self.column
            
            # Newlines
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, char, line, column))
                self.advance()
            
            # Numbers
            elif char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, line, column))
            
            # Strings
            elif char == '"':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, line, column))
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                value = self.read_identifier()
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, line, column))
            
            # Two-character operators
            elif char in '+-=!><&|':
                next_char = self.peek_char()
                two_char = char + (next_char or '')
                
                if two_char in self.operators:
                    self.tokens.append(Token(self.operators[two_char], two_char, line, column))
                    self.advance()
                    self.advance()
                elif char in self.operators:
                    self.tokens.append(Token(self.operators[char], char, line, column))
                    self.advance()
                else:
                    raise SyntaxError(f"Unknown character '{char}' at line {line}, column {column}")
            
            # Single-character operators and delimiters
            elif char in '*/(){};\n':
                if char == '(':
                    token_type = TokenType.LPAREN
                elif char == ')':
                    token_type = TokenType.RPAREN
                elif char == '{':
                    token_type = TokenType.LBRACE
                elif char == '}':
                    token_type = TokenType.RBRACE
                elif char == ';':
                    token_type = TokenType.SEMICOLON
                else:
                    token_type = self.operators.get(char)
                
                if token_type:
                    self.tokens.append(Token(token_type, char, line, column))
                self.advance()
            
            else:
                raise SyntaxError(f"Unknown character '{char}' at line {line}, column {column}")
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
