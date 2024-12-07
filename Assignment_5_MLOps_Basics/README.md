# MNIST MLOps Project

This project implements a lightweight CNN model for MNIST digit classification with MLOps practices. The model is designed to achieve >95% accuracy in one epoch while keeping parameters under 25,000.

## CI/CD Pipeline
The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs tests to verify:
   - Model has <25K parameters
   - Achieves >95% accuracy
5. Uploads trained model as artifact

## Model Architecture
- Input: 28x28 MNIST images
- 2 Convolutional layers
- MaxPooling
- Fully connected layer
- Output: 10 classes

## Testing
Tests verify:
1. Model parameter count (<25K)
2. Input/output shapes
3. Model accuracy (>95%)


# Summary of CI/CD in Machine Learning (ML)
![CICD Pipeline](artifacts/image-2.png)

## 1. CI/CD Pipeline in ML
- Continuous Integration (CI) and Continuous Deployment (CD) principles, often used in DevOps, can be adapted to streamline ML model development and deployment in production.

![CI Pipeline](artifacts/image.png)
## 2. Continuous Integration (CI)
- **Definition**: A practice where developers frequently integrate code changes into a shared repository, automating the processes of building code, resolving dependencies, and running tests.
- **Benefits**:
  - Catches bugs early, reducing integration issues.
  - Encourages smaller, more frequent code commits, simplifying code management.
  - Improves reliability in delivering code updates.

![CD Pipeline](artifacts/image-1.png)

## 3. Continuous Deployment (CD)
- **Definition**: Automates the release of software updates to production environments, eliminating manual deployment steps.
- **Benefits**:
  - Speeds up and stabilizes the release process.
  - Reduces time to market for new features.
  - Minimizes manual errors, improving release stability.

## 4. Typical Project Workflow
- Developers work on branches like development, staging, and main/master.
- Pull requests are used to merge tested and approved code into the main branch, ensuring production readiness.

## 5. CI/CD Workflow Stages
- **Code Commit**: Developers push changes to a shared repository (e.g., GitHub, GitLab).
- **Build Process**: Automated builds compile code, check for errors, and run unit tests to maintain quality.
- **Testing**: Automated tests detect integration issues; failures are flagged for developers.
- **Deployment**: Approved builds are deployed to staging or production environments.
- **Monitoring**: Systems monitor deployed changes for performance and issues.
- **Rollback Mechanism**: Provides a way to revert to previous versions if deployment issues arise.

## 6. Common CI/CD Tools
- **GitHub Actions**: Integrated solution for GitHub repositories.
- **Jenkins**: Popular open-source CI/CD tool.
- **Bitbucket Pipelines**: Tool for Bitbucket repositories.
- **AWS CodePipeline**: CI/CD service from AWS.
- **Azure DevOps**: Microsoft’s CI/CD platform.
- **GitLab CI/CD**: CI/CD tool integrated with GitLab.

## 7. MLOps and CI/CD
- **Definition**: MLOps integrates ML workflows with DevOps practices, automating the lifecycle of ML models, including building, testing, deploying, monitoring, and managing performance.
- **Key Aspects of MLOps**:
  - Model training, testing, packaging, deployment, performance monitoring, and handling issues like model drift.
- These practices are increasingly vital for data scientists and ML engineers to enhance their skillsets and career opportunities.

# Model Monitoring in CI/CD for Machine Learning (ML)

## 1. Importance of Monitoring in MLOps
- Monitoring is a critical component of the CI/CD pipeline in Machine Learning Operations (MLOps).  
- It is particularly essential for addressing **model drift**, a phenomenon where a model's performance degrades over time due to changes in the production data.

## 2. Understanding Model Drift
- **Definition**: Model drift occurs when the data used to train the model no longer reflects the current data encountered in production.  
- **Causes**:
  - Changes in user behavior.
  - Modifications in data collection methods.
  - Shifts in the underlying phenomenon being modeled.  
- **Impact**: Leads to reduced model accuracy and reliability.

## 3. Role of Monitoring
- **Purpose**: Continuously track the model's performance in production to detect issues like model drift.  
- **Metrics to Monitor**:
  - Accuracy, precision, recall, or other task-specific custom metrics.  
- **Response to Performance Drop**:
  - A significant decline in performance may signal the need for model retraining with updated data.

## 4. Addressing Model Drift
- **Detection**: Use monitoring tools and techniques to identify model drift.  
- **Retraining**: Automate the retraining process as part of the CI/CD pipeline, ensuring the model stays updated and performs optimally.

## 5. Common Monitoring Practices
- **Performance Monitoring Dashboards**:
  - Dashboards visualize key performance metrics over time.
  - Help identify trends and anomalies indicating performance degradation.
- **Automated Alerts**:
  - Alerts trigger when performance metrics fall below set thresholds or deviate significantly from expected behavior.
  - Notifications enable quick investigation and corrective action.
- **Data Drift Detection**:
  - Employ statistical techniques to monitor changes in the input feature distributions.
  - Significant deviations suggest the model might not be receiving data similar to its training data.
- **Concept Drift Detection**:
  - Analyze the relationship between input features and target variables.
  - Identify patterns in predictions that deviate from training data expectations.

## 6. Tools and Techniques
- Specific tools and methods for monitoring vary based on:
  - The ML use case.
  - Model complexity.
  - Deployment infrastructure.
- Examples include setting up custom dashboards, configuring alerts, and applying drift detection techniques.

Monitoring ensures that ML models remain accurate, reliable, and effective in dynamic production environments.

## License
MIT

## Author
[Minakshi Mathpal]