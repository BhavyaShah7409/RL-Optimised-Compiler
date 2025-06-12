from typing import List, Optional
from .lexer import Token, TokenType, Lexer
from .ast_nodes import *

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def current_token(self) -> Token:
        if self.position >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.position]
    
    def peek_token(self, offset: int = 1) -> Token:
        peek_pos = self.position + offset
        if peek_pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[peek_pos]
    
    def advance(self):
        if self.position < len(self.tokens) - 1:
            self.position += 1
    
    def match(self, *token_types: TokenType) -> bool:
        return self.current_token().type in token_types
    
    def consume(self, token_type: TokenType, message: str = "") -> Token:
        if self.current_token().type == token_type:
            token = self.current_token()
            self.advance()
            return token
        else:
            error_msg = message or f"Expected {token_type}, got {self.current_token().type}"
            raise ParseError(f"{error_msg} at line {self.current_token().line}")
    
    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def parse(self) -> Program:
        statements = []
        self.skip_newlines()
        
        while not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        return Program(statements)
    
    def parse_statement(self) -> Optional[Statement]:
        self.skip_newlines()
        
        if self.match(TokenType.PRINT):
            return self.parse_print_statement()
        elif self.match(TokenType.SCAN):
            return self.parse_scan_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.IDENTIFIER):
            return self.parse_assignment_statement()
        else:
            if not self.match(TokenType.EOF, TokenType.NEWLINE):
                raise ParseError(f"Unexpected token {self.current_token().type} at line {self.current_token().line}")
            return None
    
    def parse_print_statement(self) -> PrintStatement:
        self.consume(TokenType.PRINT)
        self.consume(TokenType.LPAREN)
        expr = self.parse_expression()
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.SEMICOLON)
        return PrintStatement(expr)
    
    def parse_scan_statement(self) -> ScanStatement:
        self.consume(TokenType.SCAN)
        self.consume(TokenType.LPAREN)
        target = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.SEMICOLON)
        return ScanStatement(target)
    
    def parse_assignment_statement(self) -> AssignmentStatement:
        target = self.consume(TokenType.IDENTIFIER).value
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN):
            operator = self.current_token().value
            self.advance()
            value = self.parse_expression()
            self.consume(TokenType.SEMICOLON)
            return AssignmentStatement(target, operator, value)
        else:
            raise ParseError(f"Expected assignment operator at line {self.current_token().line}")
    
    def parse_while_statement(self) -> WhileStatement:
        self.consume(TokenType.WHILE)
        self.consume(TokenType.LPAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        return WhileStatement(condition, body)
    
    def parse_if_statement(self) -> IfStatement:
        self.consume(TokenType.IF)
        self.consume(TokenType.LPAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        then_body = []
        while not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                then_body.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        else_body = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.consume(TokenType.LBRACE)
            self.skip_newlines()
            
            else_body = []
            while not self.match(TokenType.RBRACE) and not self.match(TokenType.EOF):
                stmt = self.parse_statement()
                if stmt:
                    else_body.append(stmt)
                self.skip_newlines()
            
            self.consume(TokenType.RBRACE)
        
        return IfStatement(condition, then_body, else_body)
    
    def parse_expression(self) -> Expression:
        return self.parse_logical_or()
    
    def parse_logical_or(self) -> Expression:
        expr = self.parse_logical_and()
        
        while self.match(TokenType.OR):
            operator = self.current_token().value
            self.advance()
            right = self.parse_logical_and()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_logical_and(self) -> Expression:
        expr = self.parse_equality()
        
        while self.match(TokenType.AND):
            operator = self.current_token().value
            self.advance()
            right = self.parse_equality()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_equality(self) -> Expression:
        expr = self.parse_comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.current_token().value
            self.advance()
            right = self.parse_comparison()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_comparison(self) -> Expression:
        expr = self.parse_addition()
        
        while self.match(TokenType.GREATER, TokenType.LESS, TokenType.GREATER_EQUAL, TokenType.LESS_EQUAL):
            operator = self.current_token().value
            self.advance()
            right = self.parse_addition()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_addition(self) -> Expression:
        expr = self.parse_multiplication()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.current_token().value
            self.advance()
            right = self.parse_multiplication()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_multiplication(self) -> Expression:
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE):
            operator = self.current_token().value
            self.advance()
            right = self.parse_unary()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def parse_unary(self) -> Expression:
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.current_token().value
            self.advance()
            operand = self.parse_unary()
            return UnaryOperation(operator, operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> Expression:
        if self.match(TokenType.NUMBER):
            value = float(self.current_token().value)
            self.advance()
            return NumberLiteral(value)
        
        elif self.match(TokenType.STRING):
            value = self.current_token().value
            self.advance()
            return StringLiteral(value)
        
        elif self.match(TokenType.IDENTIFIER):
            name = self.current_token().value
            self.advance()
            return Identifier(name)
        
        elif self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN)
            return expr
        
        else:
            raise ParseError(f"Unexpected token {self.current_token().type} at line {self.current_token().line}")
