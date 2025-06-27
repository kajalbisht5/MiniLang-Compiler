import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# === Token ===
class Token:
    def __init__(self, type_, value, line=None, column=None):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def to_dict(self):
        return {"type": self.type, "value": self.value, "line": self.line, "column": self.column}

    def __repr__(self):
        return f"Token({self.type}, '{self.value}', Line: {self.line}, Col: {self.column})"


# === Lexer ===
class Lexer:
    def __init__(self):
        self.token_spec = [
            ('INCLUDE', r'#\s*include\s*<[^>]+>'),
            # Added scanf, more comprehensive keywords might be needed for full C/C++
            ('KEYWORD', r'\b(int|float|char|double|long|short|void|return|if|else|for|while|do|printf|scanf|main)\b'),
            ('ID',      r'\b[a-zA-Z_]\w*\b'),
            ('NUM',     r'\b\d+(\.\d+)?\b'),
            # Improved regex for strings to handle escaped quotes
            ('STRING',  r'"([^"\\]|\\.)*"'), 
            ('RELOP',   r'(==|!=|<=|>=|<|>)'), # Relational Operators
            ('OP',      r'[=+\-*/&!]'), # Arithmetic, Assignment, and Address-of Operator
            ('LPAREN',  r'\('),
            ('RPAREN',  r'\)'),
            ('LBRACE',  r'\{'),
            ('RBRACE',  r'\}'),
            ('LBRACKET',r'\['),       
            ('RBRACKET',r'\]'),         
            ('DELIM',   r'[;,]'),
            ('COMMENT', r'//.*|/\*[\s\S]*?\*/'), # Single and multi-line comments
            ('SKIP',    r'[ \t\n]+'),
            ('MISMATCH', r'.'), # Any other character
        ]
        self.pattern = re.compile('|'.join(f'(?P<{name}>{regex})' for name, regex in self.token_spec))

    def tokenize(self, code):
        tokens = []
        line_num = 1
        col_num = 1

        for mo in self.pattern.finditer(code):
            kind = mo.lastgroup
            value = mo.group()

            current_len = len(value)
            
            # Update line and column numbers
            if '\n' in value:
                lines_in_value = value.count('\n')
                line_num += lines_in_value
                col_num = current_len - value.rfind('\n') 
            else:
                col_num += current_len

            if kind == 'SKIP':
                # Only update column for whitespace that doesn't include newlines
                pass
            elif kind == 'COMMENT':
                pass # Comments are skipped, line/col updated by the general logic
            elif kind == 'MISMATCH':
                raise RuntimeError(f"Lexical Error: Unexpected character '{value}' at Line {line_num}, Column {col_num - current_len}")
            else:
                tokens.append(Token(kind, value, line_num, col_num - current_len)) # Store start column
                
        return tokens


# === Parser ===
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.ast = []
        self.errors = []

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self, offset=1):
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def match(self, type_, value=None):
        token = self.current()
        if token and token.type == type_ and (value is None or token.value == value):
            self.pos += 1
            return token
        return None

    def error(self, message):
        token = self.current()
        location = f"Line {token.line}, Column {token.column}" if token else "End of file"
        self.errors.append(f"Syntax Error: {message} at {location}")
        # Simple error recovery: advance past the current token to prevent infinite loops
        if token:
            self.pos += 1

    def parse(self):
        while self.pos < len(self.tokens):
            token = self.current()
            if not token: # Reached end of tokens during parsing
                break

            if token.type == 'INCLUDE':
                self.ast.append(('include', token.value))
                self.pos += 1
            elif token.type == 'KEYWORD' and token.value in {'int', 'float', 'char', 'void', 'double', 'long', 'short'}:
                if self.lookahead_is_function():
                    self.function_definition()
                else:
                    self.parse_declaration_or_assignment()
            elif token.type == 'KEYWORD' and token.value == 'if':
                self.if_statement()
            elif token.type == 'KEYWORD' and token.value == 'printf':
                self.printf_statement()
            elif token.type == 'KEYWORD' and token.value == 'scanf':
                self.scanf_statement()
            elif token.type == 'KEYWORD' and token.value == 'return':
                self.return_statement()
            elif token.type == 'ID': # For assignments to already declared variables
                 self.parse_assignment()
            else:
                self.error(f"Unexpected token '{token.value}'")
                # Attempt to skip to the next potential statement start or end of block
                while self.current() and self.current().type not in {'KEYWORD', 'INCLUDE', 'ID', 'RBRACE', 'DELIM'}:
                    self.pos += 1
                if self.current() and self.current().type == 'DELIM': # Consume trailing semicolon
                    self.pos += 1
        return self.ast, self.errors

    def lookahead_is_function(self):
        # Checks for "TYPE ID (" or "TYPE main ("
        temp_pos = self.pos
        if self.match('KEYWORD'): # Consume type (e.g., 'int')
            next_token = self.current()
            if next_token and next_token.type in {'ID', 'KEYWORD'}:
                if self.peek() and self.peek().type == 'LPAREN':
                    self.pos = temp_pos # Reset pos
                    return True
        self.pos = temp_pos # Reset pos
        return False

    def function_definition(self):
        ret_type = self.match('KEYWORD') # int, void etc.
        func_name = self.match('ID') or self.match('KEYWORD', 'main')
        
        if not (ret_type and func_name):
            self.error("Invalid function definition start")
            # Try to recover by skipping to next brace or semicolon
            while self.current() and self.current().type not in {'LBRACE', 'DELIM'}:
                self.pos += 1
            return

        if not self.match('LPAREN'):
            self.error(f"Expected '(' after function name '{func_name.value}'")
            return
        
        # Simple handling for parameters (just skip them for now)
        # To parse properly, you'd need parameter_list production
        while self.current() and self.current().type != 'RPAREN':
            self.pos += 1
            if not self.current(): # Handle EOF
                self.error("Unexpected end of file in function parameters")
                return

        if not self.match('RPAREN'):
            self.error("Expected ')' in function signature")
            return

        if not self.match('LBRACE'):
            self.error("Missing '{' for function body")
            return

        body = []
        while self.current() and self.current().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                # If parse_statement returns None, it means an error occurred, advance to avoid loop
                if self.pos < len(self.tokens):
                    self.pos += 1 # Advance to try next token as statement start

        if not self.match('RBRACE'):
            self.error("Missing '}' in function body")
        
        self.ast.append(('function_def', ret_type.value, func_name.value, body))

    def parse_statement(self):
        token = self.current()
        if not token:
            return None # End of file

        if token.type == 'KEYWORD':
            if token.value in {'int', 'float', 'char', 'double', 'long', 'short'}:
                return self.parse_declaration_or_assignment()
            elif token.value == 'printf':
                return self.printf_statement()
            elif token.value == 'scanf':
                return self.scanf_statement()
            elif token.value == 'return':
                return self.return_statement()
            elif token.value == 'if':
                return self.if_statement()
            # Add more statement types here (while, for, etc.)
        elif token.type == 'ID':
            return self.parse_assignment()
        
        # If none of the above, it's an unexpected token at statement start
        self.error(f"Unexpected token '{token.value}' at start of statement")
        return None

    def parse_declaration_or_assignment(self):
        start_pos = self.pos
        type_tok = self.match('KEYWORD') # int, float etc.
        id_tok = self.match('ID')

        if not (type_tok and id_tok):
            self.error("Invalid declaration or assignment start")
            # Attempt to skip to next semicolon or brace for recovery
            while self.current() and self.current().type not in {'DELIM', 'RBRACE'}:
                self.pos += 1
            if self.current() and self.current().type == 'DELIM': self.pos += 1
            return None
        
        node = None
        if self.match('LBRACKET'): # Array declaration
            size_tok = self.match('NUM')
            if not size_tok:
                self.error("Expected array size")
                return None
            if not self.match('RBRACKET'):
                self.error("Expected ']' in array declaration")
                return None
            node = ('declare_array', type_tok.value, id_tok.value, size_tok.value)
        else: # Simple variable declaration
            node = ('declare', type_tok.value, id_tok.value)
        
        if self.match('OP', '='): # Assignment part
            expr = self.expression()
            if not expr:
                self.error("Invalid expression for assignment")
                return None
            # For combined declaration and assignment, we store it as an assignment
            # Semantic analyzer will ensure it's declared if this is the first time.
            # If it's a redeclaration, semantic analyzer will catch it.
            node = ('assign_expr', id_tok.value, expr) 
        
        if not self.match('DELIM'):
            self.error("Missing semicolon at end of declaration/assignment")
            # Attempt to synchronize by skipping to next semicolon or brace
            while self.current() and self.current().type not in {'DELIM', 'RBRACE'}:
                self.pos += 1
            if self.current() and self.current().type == 'DELIM':
                self.pos += 1 # Consume the semicolon
        
        self.ast.append(node)
        return node


    def parse_assignment(self):
        id_tok = self.match('ID')
        if not id_tok:
            self.error("Expected identifier for assignment")
            return None
        
        if not self.match('OP', '='):
            self.error("Expected '=' for assignment")
            # Attempt to recover by skipping to semicolon
            while self.current() and self.current().type not in {'DELIM', 'RBRACE'}:
                self.pos += 1
            if self.current() and self.current().type == 'DELIM': self.pos += 1
            return None
        
        expr = self.expression()
        if not expr:
            self.error("Invalid expression for assignment")
            return None
        
        if not self.match('DELIM'):
            self.error("Missing semicolon at end of assignment")
            # Attempt to synchronize
            while self.current() and self.current().type not in {'DELIM', 'RBRACE'}:
                self.pos += 1
            if self.current() and self.current().type == 'DELIM':
                self.pos += 1 # Consume the semicolon

        node = ('assign_expr', id_tok.value, expr)
        self.ast.append(node)
        return node

    def printf_statement(self):
        start_pos = self.pos
        if not self.match('KEYWORD', 'printf'):
            return None # Not a printf statement

        if not self.match('LPAREN'):
            self.error("Expected '(' after printf")
            return None
        
        args = []
        str_tok = self.match('STRING')
        if not str_tok:
            self.error("Expected format string in printf")
            # Try to recover by skipping to ')' or ';'
            while self.current() and self.current().type not in {'RPAREN', 'DELIM'}:
                self.pos += 1
            # Don't return, try to consume ')' and ';'
        else:
            args.append(str_tok.value)
            while self.match('DELIM', ','): # For additional arguments
                expr_arg = self.expression()
                if expr_arg:
                    args.append(expr_arg)
                else:
                    self.error("Expected expression after comma in printf")
                    break

        if not self.match('RPAREN'):
            self.error("Expected ')' in printf statement")
            # Recovery: try to find semicolon
            while self.current() and self.current().type != 'DELIM':
                self.pos += 1
        
        if not self.match('DELIM'):
            self.error("Missing semicolon after printf statement")
            # Recovery: try to find next statement start
            while self.current() and self.current().type not in {'KEYWORD', 'ID', 'RBRACE'}:
                self.pos += 1
        
        node = ('printf', args)
        self.ast.append(node)
        return node

    def scanf_statement(self):
        if not self.match('KEYWORD', 'scanf'):
            return None
        
        if not self.match('LPAREN'):
            self.error("Expected '(' after scanf")
            return None
        
        args = []
        format_str_tok = self.match('STRING')
        if not format_str_tok:
            self.error("Expected format string in scanf")
        else:
            args.append(format_str_tok.value)

        # Parse subsequent arguments
        while self.match('DELIM', ','):
            # Expecting '&' followed by an ID
            if not self.match('OP', '&'):
                self.error("Expected '&' before variable in scanf argument")
                # Recovery: skip to next comma or ')'
                while self.current() and self.current().type not in {'DELIM', 'RPAREN'}:
                    self.pos += 1
                continue

            var_tok = self.match('ID')
            if not var_tok:
                self.error("Expected variable name after '&' in scanf argument")
                # Recovery: skip to next comma or ')'
                while self.current() and self.current().type not in {'DELIM', 'RPAREN'}:
                    self.pos += 1
            else:
                args.append(('address_of', var_tok.value))
        
        if not self.match('RPAREN'):
            self.error("Expected ')' in scanf statement")
        
        if not self.match('DELIM'):
            self.error("Missing semicolon after scanf statement")
            # Recovery: skip to next statement start
            while self.current() and self.current().type not in {'KEYWORD', 'ID', 'RBRACE'}:
                self.pos += 1

        node = ('scanf', args)
        self.ast.append(node)
        return node


    def return_statement(self):
        if not self.match('KEYWORD', 'return'):
            return None
        
        expr = None
        # Check if there's an expression before semicolon
        if self.current() and self.current().type != 'DELIM':
            expr = self.expression() # return expression;
        
        if not self.match('DELIM'):
            self.error("Missing semicolon after return statement")
        
        node = ('return', expr)
        self.ast.append(node)
        return node

    def if_statement(self):
        if not self.match('KEYWORD', 'if'):
            return None
        
        if not self.match('LPAREN'):
            self.error("Expected '(' after 'if'")
            return None
        
        condition = self.expression()
        if not condition:
            self.error("Expected condition in 'if' statement")
            return None
        
        if not self.match('RPAREN'):
            self.error("Expected ')' after 'if' condition")
            return None
        
        if_body = []
        if self.match('LBRACE'):
            while self.current() and self.current().type != 'RBRACE':
                stmt = self.parse_statement()
                if stmt:
                    if_body.append(stmt)
                else: # Error recovery within block
                    if self.pos < len(self.tokens):
                        self.pos += 1 # Advance to try next token
            if not self.match('RBRACE'):
                self.error("Expected '}' for 'if' block")
        else: # Single statement without braces
            stmt = self.parse_statement()
            if stmt:
                if_body.append(stmt)
            else:
                self.error("Expected statement after 'if' condition")
        
        else_body = []
        if self.match('KEYWORD', 'else'):
            if self.match('LBRACE'):
                while self.current() and self.current().type != 'RBRACE':
                    stmt = self.parse_statement()
                    if stmt:
                        else_body.append(stmt)
                    else: # Error recovery within block
                        if self.pos < len(self.tokens):
                            self.pos += 1 # Advance to try next token
                if not self.match('RBRACE'):
                    self.error("Expected '}' for 'else' block")
            else: # Single statement without braces
                stmt = self.parse_statement()
                if stmt:
                    else_body.append(stmt)
                else:
                    self.error("Expected statement after 'else'")

        node = ('if_else', condition, if_body, else_body if else_body else None)
        self.ast.append(node)
        return node

    def expression(self):
        return self.relational_expression()

    def relational_expression(self):
        node = self.add_sub()
        while True:
            op_tok = self.match('RELOP')
            if op_tok:
                right = self.add_sub()
                if not right:
                    self.error("Expected expression after relational operator")
                    return None
                node = ('binop', op_tok.value, node, right)
            else:
                break
        return node

    def add_sub(self):
        node = self.mul_div()
        while True:
            op_tok = self.match('OP', '+') or self.match('OP', '-')
            if op_tok:
                right = self.mul_div()
                if not right:
                    self.error("Expected expression after arithmetic operator")
                    return None
                node = ('binop', op_tok.value, node, right)
            else:
                break
        return node

    def mul_div(self):
        node = self.factor()
        while True:
            op_tok = self.match('OP', '*') or self.match('OP', '/')
            if op_tok:
                right = self.factor()
                if not right:
                    self.error("Expected expression after arithmetic operator")
                    return None
                node = ('binop', op_tok.value, node, right)
            else:
                break
        return node

    def factor(self):
        tok = self.current()
        if not tok:
            return None # End of tokens

        if tok.type == 'NUM':
            self.pos += 1
            return ('literal', tok.value)
        elif tok.type == 'ID':
            self.pos += 1
            return ('variable', tok.value)
        elif tok.type == 'LPAREN':
            self.match('LPAREN')
            expr = self.expression()
            if not expr:
                self.error("Expected expression inside parentheses")
                return None
            if not self.match('RPAREN'):
                self.error("Expected ')' after expression")
            return expr
        
        self.error(f"Expected number, identifier, or '(' but got '{tok.value}'")
        return None


# === Semantic Analyzer ===
class SemanticAnalyzer:
    def __init__(self, ast):
        self.ast = ast
        self.symbols = {} # Stores {'name': {'type': 'int', 'scope': 'global/main'}}
        self.errors = []
        self.current_scope_name = 'global' # Tracks the current scope (e.g., 'global', 'main')
        self.scopes = {'global': {}} # A dictionary to hold symbols for each scope

    def analyze(self):
        for node in self.ast:
            self._analyze_node(node)
        return self.errors

    def _add_symbol(self, name, symbol_type):
        if name in self.scopes[self.current_scope_name]:
            self.errors.append(f"Semantic Error: '{name}' redeclared in '{self.current_scope_name}' scope.")
        else:
            self.scopes[self.current_scope_name][name] = {'type': symbol_type, 'scope': self.current_scope_name}

    def _get_symbol(self, name):
        # Check current scope first, then global
        if name in self.scopes[self.current_scope_name]:
            return self.scopes[self.current_scope_name][name]
        if self.current_scope_name != 'global' and name in self.scopes['global']:
            return self.scopes['global'][name]
        return None

    def _analyze_node(self, node):
        node_type = node[0]

        if node_type == 'function_def':
            _, ret_type, func_name, body = node
            if func_name in self.scopes['global']: # Functions are always global for now
                self.errors.append(f"Semantic Error: Function '{func_name}' redeclared.")
            else:
                self.scopes['global'][func_name] = {'type': 'function', 'return_type': ret_type, 'scope': 'global'}
            
            # Enter function scope
            previous_scope_name = self.current_scope_name
            self.current_scope_name = func_name
            self.scopes[func_name] = {} # Create new scope for function locals

            for stmt in body:
                self._analyze_node(stmt)
            
            # Exit function scope
            self.current_scope_name = previous_scope_name 

        elif node_type == 'declare':
            _, var_type, var_name = node
            self._add_symbol(var_name, var_type)
        
        elif node_type == 'declare_array':
            _, var_type, var_name, size = node
            self._add_symbol(var_name, f"array_of_{var_type}")

        elif node_type == 'assign_expr':
            var_name, expr_node = node[1], node[2] 
            symbol_info = self._get_symbol(var_name)
            if not symbol_info:
                self.errors.append(f"Semantic Error: Undeclared variable '{var_name}' used in assignment.")
            # TODO: Add type compatibility check between assigned value and variable type
            self._check_expression_types(expr_node)

        elif node_type == 'printf':
            # Args are at index 1 of the printf tuple
            for arg in node[1]:
                if isinstance(arg, tuple) and arg[0] in {'variable', 'binop', 'literal'}: 
                    self._check_expression_types(arg)

        elif node_type == 'scanf':
            # Scanf arguments: first is format string, subsequent are ('address_of', var_name)
            for i, arg in enumerate(node[1]):
                if i == 0: continue # Skip format string for now
                
                if arg[0] == 'address_of':
                    var_name = arg[1]
                    symbol_info = self._get_symbol(var_name)
                    if not symbol_info:
                        self.errors.append(f"Semantic Error: Undeclared variable '{var_name}' used in scanf.")
                # TODO: Match format specifiers with variable types

        elif node_type == 'return':
            expr = node[1]
            if expr:
                self._check_expression_types(expr)
            # TODO: Add check for return type matching function's declared return type

        elif node_type == 'if_else':
            condition, if_body, else_body = node[1], node[2], node[3]
            self._check_expression_types(condition) # Ensure condition is evaluate-able
            for stmt in if_body:
                self._analyze_node(stmt)
            if else_body:
                for stmt in else_body:
                    self._analyze_node(stmt)
        
        # Add analysis for other node types as they are added to the AST

    def _check_expression_types(self, expr_node):
        if not expr_node:
            return

        if expr_node[0] == 'literal':
            # TODO: Determine literal type (int, float, char, string)
            pass
        elif expr_node[0] == 'variable':
            var_name = expr_node[1]
            symbol_info = self._get_symbol(var_name)
            if not symbol_info:
                self.errors.append(f"Semantic Error: Undeclared variable '{var_name}' used in expression.")
            # TODO: Further type checking: e.g., ensure arithmetic ops are on numeric types
        elif expr_node[0] == 'binop':
            op, left, right = expr_node[1], expr_node[2], expr_node[3]
            self._check_expression_types(left)
            self._check_expression_types(right)
            # TODO: Add rules for operator type compatibility (e.g., cannot add int to string)


# === Intermediate Code Generator ===
class ICG:
    def __init__(self, ast):
        self.ast = ast
        self.code = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self, prefix='L'):
        self.label_count += 1
        return f"{prefix}{self.label_count}"

    def generate(self):
        for stmt in self.ast:
            self._generate_node(stmt)
        return self.code

    def _generate_node(self, node):
        node_type = node[0]

        if node_type == 'include':
            self.code.append(node[1])
        elif node_type == 'declare':
            _, typ, name = node
            self.code.append(f"DECLARE {typ} {name}")
        elif node_type == 'declare_array':
            _, typ, name, size = node
            self.code.append(f"DECLARE_ARRAY {typ} {name}[{size}]")
        elif node_type == 'assign_expr':
            var_name, expr = node[1], node[2] 
            result = self._handle_expr(expr)
            self.code.append(f"ASSIGN {var_name}, {result}")
        elif node_type == 'printf':
            args_icg = []
            for arg in node[1]:
                if isinstance(arg, str): # Raw string literal (format string)
                    args_icg.append(arg)
                else: # It's an expression
                    args_icg.append(self._handle_expr(arg))
            self.code.append(f"CALL printf, {', '.join(args_icg)}")
        elif node_type == 'scanf':
            # Scanf arguments: first is format string, subsequent are ('address_of', var_name)
            args_icg = []
            for arg in node[1]:
                if isinstance(arg, str): # Format string
                    args_icg.append(arg)
                elif arg[0] == 'address_of': # Variable reference
                    args_icg.append(f"&{arg[1]}")
            self.code.append(f"CALL scanf, {', '.join(args_icg)}")
        elif node_type == 'return':
            expr_val = self._handle_expr(node[1]) if node[1] else "VOID"
            self.code.append(f"RETURN {expr_val}")
        elif node_type == 'function_def':
            _, ret_type, func_name, body = node
            self.code.append(f"\nFUNCTION {ret_type} {func_name}():")
            for stmt in body:
                self._generate_node(stmt)
            self.code.append(f"END_FUNCTION {func_name}\n")
        elif node_type == 'if_else':
            condition, if_body, else_body = node[1], node[2], node[3]
            
            cond_result = self._handle_expr(condition)
            else_label = self.new_label("ELSE")
            end_label = self.new_label("ENDIF")

            self.code.append(f"IF_FALSE {cond_result} GOTO {else_label}")
            for stmt in if_body:
                self._generate_node(stmt)
            self.code.append(f"GOTO {end_label}")

            self.code.append(f"{else_label}:")
            if else_body:
                for stmt in else_body:
                    self._generate_node(stmt)
            self.code.append(f"{end_label}:")

    def _handle_expr(self, node):
        if not node:
            return "NULL" 

        if node[0] == 'literal':
            return node[1]
        elif node[0] == 'variable':
            return node[1]
        elif node[0] == 'binop':
            _, op, left, right = node
            l = self._handle_expr(left)
            r = self._handle_expr(right)
            temp = self.new_temp()
            self.code.append(f"{temp} = {l} {op} {r}")
            return temp
        
        return "UNKNOWN_EXPR"


# === Flask Routes ===
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    code = request.json.get('code', '')

    try:
        lexer = Lexer()
        tokens = lexer.tokenize(code)

        parser = Parser(tokens)
        ast, syntax_errors = parser.parse()

        semantic = SemanticAnalyzer(ast)
        semantic_errors = semantic.analyze()

        icg = ICG(ast)
        intermediate_code = icg.generate()

        return jsonify({
            "tokens": [t.to_dict() for t in tokens],
            "ast": ast, # Include AST for debugging/visualization
            "syntaxErrors": syntax_errors,
            "semanticErrors": semantic_errors,
            "intermediateCode": intermediate_code
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)