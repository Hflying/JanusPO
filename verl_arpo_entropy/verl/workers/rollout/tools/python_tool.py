import ast
import subprocess
from typing import Tuple

from verl.workers.rollout.tools.base_tool import BaseTool


class PythonTool(BaseTool):
    """Python conda """
    
    def __init__(self, conda_path: str, conda_env: str):
        """
         Python 
        
        Args:
            conda_path: conda 
            conda_env: conda 
        """
        self.conda_path = conda_path
        self.conda_env = conda_env
        self.python_path = f"{conda_path}/envs/{conda_env}/bin/python"

    @property
    def name(self) -> str:
        return "python_interpreter"
    
    @property
    def trigger_tag(self) -> str:
        return "python"
    
    def execute(self, code: str, timeout: int = 120) -> str:
        """ Python """
        result, report = self._run_code(code, timeout)
        
        if report == "Done":
            return result
        else:
            return report
    
    def _run_code(self, code: str, timeout: int) -> Tuple[str, str]:
        """ conda Python """

        code = self._preprocess_code(code)
        
        try:
            #   subprocess.run  
            process = subprocess.run(
                [self.python_path, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False 
            )

            if process.returncode == 0:
                return process.stdout.strip(), "Done"
            else:
                return "", process.stderr.strip()

        except subprocess.TimeoutExpired:
            return "", f"  {timeout}  "
        except Exception as e:
            return "", f" : {str(e)}"
    
    def _preprocess_code(self, code: str) -> str:
        """
         Python 
         print print 
        """
        try:
            tree = ast.parse(code)
            if tree.body:
                last_expr = tree.body[-1]
                if isinstance(last_expr, ast.Expr):
                    #  print 
                    if not (isinstance(last_expr.value, ast.Call) 
                            and isinstance(last_expr.value.func, ast.Name) 
                            and last_expr.value.func.id == 'print'):
                        print_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='print', ctx=ast.Load()),
                                args=[last_expr.value],
                                keywords=[]
                            )
                        )
                        tree.body[-1] = print_call
                        code = ast.unparse(tree)
        except:
            pass  #  
        
        return code

def _test():
    batch_code = [
        """

x = sympy.symbols('x')
y = sympy.symbols('y')


expr = x**2 + 2*x*y + y**2

print(f"Expression: {expr}")


derivative = sympy.diff(expr, x)
print(f"Derivative with respect to x: {derivative}")


result = expr.subs([(x, 1), (y, 2)])
print(f"Value at x=1, y=2: {result}")
        """,
        """
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        print(np.array([1, 2, 3]))
        """
    ]
    
    async def run_test():
        #  Python 
        python_tool = PythonTool(
            conda_path="/mmu_nlp_ssd/makai05/miniconda3/",  #  conda 
            conda_env="verl",              #  
            max_concurrent=64
        )
        

        for i, code in enumerate(batch_code):
            print(f"\n---   {i+1} ---")
            result = await python_tool.execute(code)
            print(f" :\n{result}")
    

    asyncio.run(run_test())

if __name__ == "__main__":
    _test()


