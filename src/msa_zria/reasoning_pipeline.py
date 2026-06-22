

# Define modules
parser = ParseModule()
code_gen = CodeGenModule()  # similar to ParseModule, outputs a CodeOutput (string with code)
evaluator = EvalModule()

# Compose workflow: Parse -> Code -> Run Pyro -> Evaluate
# Using a custom Orchestrator class to tie it together in code for clarity
class ReasoningPipeline:
    def __init__(self):
        self.parser = ParseModule()
        self.code_gen = CodeGenModule()
        self.pyro_tool = PythonInterpreter()
        self.evaluator = EvalModule()
    def __call__(self, customer_query):
        parsed = self.parser(customer_query)            # structured ParseOutput
        code = self.code_gen(parsed=parsed)             # generate code using parsed info
        pyro_result = self.pyro_tool.run(code.code_str) # execute code (assuming code.code_str is the string)
        evaluation = self.evaluator(query=customer_query, answer=pyro_result)
        return evaluation
pipeline = ReasoningPipeline()
output = pipeline("User: ... (some query)...")
print(output)
