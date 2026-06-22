# DSPy imports
from dspy import Signature, InputField, OutputField, Module, Tool, PythonInterpreter


# ------------------------ DSPy Modules ------------------------
class ParseModule(Module):
    """
    Module: Parse a customer query into structured fields (device, issue).
    """
    signature = Signature(
        inputs=InputField(str, "text"),
        outputs=OutputField(dict, "parsed_result")
    )

    def prompt(self, text: str) -> str:
        return (
            "[PARSE] Extract the 'device' and 'issue' from the following customer message. "
            "Return a JSON object with keys 'device' and 'issue'.\n" +
            f"Message: {text}"
        )

class CodeGenModule(Module):
    """
    Module: Generate Pyro code that models the parsed scenario.
    """
    signature = Signature(
        inputs=InputField(dict, "parsed"),
        outputs=OutputField(str, "code_str")
    )

    def prompt(self, parsed: dict) -> str:
        device = parsed.get('device')
        issue = parsed.get('issue')
        return (
            "[CODE] Given the device and issue facts, write a Pyro probabilistic program. "
            "Use the variable 'observed' to condition on the issue.\n" +
            f"Facts: device = '{device}', issue = '{issue}'.\n"
            "Query: sample the probability of 'failure' given 'observed'."
        )

class EvalModule(Module):
    """
    Module: Evaluate the model's answer in natural language.
    """
    signature = Signature(
        inputs=[InputField(str, "query"), InputField(Any, "answer")],
        outputs=OutputField(str, "evaluation")
    )

    def prompt(self, query: str, answer: Any) -> str:
        return (
            "[EVALUATE] Given the user query and model outcome, explain whether the outcome addresses the user's issue.\n" +
            f"Query: {query}\nOutcome: {answer}"
        )
        
        # DSPy Tools
# Python interpreter to run generated code
pyro_interpreter = PythonInterpreter()

