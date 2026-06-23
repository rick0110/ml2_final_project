# Original User Request

## Initial Request — 2026-06-20T23:57:04-03:00

Analyze all files in the project directory to identify methodological and architectural Machine Learning errors. Generate a detailed report validating these findings with academic papers and authoritative web resources.

Working directory: /home/richard/project/ml2_final_project
Integrity mode: development

## Requirements

### R1. Identify Methodological and Architectural Flaws
Analyze the source code to find conceptual, methodological, and architectural errors in the Machine Learning pipeline (e.g., data leakage, incorrect evaluation metrics, suboptimal or flawed neural network architectures, poor hyperparameter choices, bad scaling strategies). Ignore simple syntax or formatting errors.

### R2. Web Validation
For every identified flaw, conduct a web search to find academic papers or authoritative references that prove why the current implementation is flawed and what the correct approach should be.

### R3. Generate Report
Produce a structured markdown file named `ml_methodology_review.md` detailing the findings. Each entry must clearly state the file path, the flawed logic, the explanation, and the external reference used for validation.

## Acceptance Criteria

### Report Structure and Quality
- [ ] The file `ml_methodology_review.md` exists in the working directory.
- [ ] Every error listed in the report specifies the exact file path and relevant code snippet or logic description.
- [ ] Every error listed is accompanied by at least one citation (academic paper title, authors, or a direct URL) validating the claim.
- [ ] The report focuses exclusively on ML methodology and architecture, avoiding generic software bugs.

## Follow-up — 2026-06-21T15:07:42-03:00

Conduct a comprehensive code review and validation of the existing project. Navigate through every file and line of code, verify the correctness of the implementation, and validate the underlying methods and approaches by researching related papers and web resources.

Working directory: /home/richard/project/ml2_final_project
Integrity mode: development

## Requirements

### R1. Comprehensive Code Review
Analyze every source code file and line of code in the project for implementation errors, logical bugs, and best practices.

### R2. Methodological Validation
For each significant method or algorithm used in the codebase, search the web and academic literature to confirm that the approach is theoretically sound, valid, and correctly applied. Include citations or URLs for validation.

### R3. Output Deliverable
Produce a comprehensive Markdown report detailing the findings. The report must contain an inventory of all files analyzed, specific errors found (or a note if none), and the validation references for the methods.

## Acceptance Criteria

### Review Completeness
- [ ] The report contains a complete checklist/inventory of all source files in the project.
- [ ] An independent agent-as-judge can verify that the report covers 100% of the project's source code files.

### Validation Depth
- [ ] Every significant method identified in the report has at least one corresponding web URL or academic paper citation validating its approach.
- [ ] The report includes a clear "Errors and Improvements" section for any implementation issues found.
