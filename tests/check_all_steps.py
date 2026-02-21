import os
import sys
import ast

def find_missing_attrs(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            with open(path, "r") as f:
                try:
                    tree = ast.parse(f.read())
                except:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it inherits from Step
                    # Or just check if name ends with 'Step'
                    if not node.name.endswith("Step"):
                        continue
                    
                    has_name = False
                    has_weight = False
                    for child in node.body:
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name):
                                    if target.id == "name": has_name = True
                                    if target.id == "weight": has_weight = True
                        elif isinstance(child, ast.AnnAssign):
                            if isinstance(child.target, ast.Name):
                                if child.target.id == "name": has_name = True
                                if child.target.id == "weight": has_weight = True
                    
                    if not has_name or not has_weight:
                        print(f"{file}: {node.name} is missing name={not has_name}, weight={not has_weight}")

if __name__ == "__main__":
    find_missing_attrs(os.path.abspath("../spectrue_core/pipeline/steps"))

