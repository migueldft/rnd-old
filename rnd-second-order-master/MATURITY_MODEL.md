# Maturity Model Checker

This document is a reference to the [Dafiti Maturity Model](https://dafiti.jira.com/wiki/spaces/DFTEC/pages/180525103/DFTech+Maturity+Model)

Follow below are what is done according with the maturity model.

## Architecture

#### 1. Basic cloud-native

- [ ] 1A - One codebase
- [ ] 1B - One CI pipeline
- [ ] 1C - API First
- [ ] 1D - Dependency management
- [ ] 1E - Design, build, release, and run
- [ ] 1F - Logs
- [ ] 1G - Environment parity
- [ ] 1H - Administrative processes
- [ ] 1I - Port binding
- [ ] 1J - Stateless processes
- [ ] 1K - Basic Telemetry
- [ ] 1L - Basic Authentication and authorization
- [ ] 1M - Documentation

#### 2. Domain driven & Advanced Cloud-native

- [ ] 2A - Bounded contexts and domains are defined and publicly available
- [ ] 2B - Configuration, credentials, and code
- [ ] 2C - Concurrency

#### 3. Expert Cloud-native

- [ ] 3A - Disposability
- [ ] 3B - Backing services
- [ ] 3C - Advanced Telemetry
- [ ] 3D - Advanced Authentication and authorization

#### 4. Legacy free

- [ ] 4A - No Bob
- [ ] 4B - High-End


## Infrastructure & Operations

#### 1. Deployment

- [ ] 1  - Automated deployment

#### 2. Basic SLA

- [ ] 2A - Basic SLA: max error rate defined in % (0,02)
- [ ] 2B - Basic SLA: min availability defined in % (99,98)
- [ ] 2C - Basic SLA: max latency defined in ms (15000)

#### 3. Runbooks

- [ ] 3  - Runbooks and recovery SLA

#### 4. Reliability
- [ ] 4  - Reliability and resilience


## Quality

#### 1. Basics

- [ ] 1A - Coding Standards are followed
- [ ] 1B - Documentation exists
- [ ] 1C - Sufficient unit tests exist defined in % (70)

#### 2. Basic Automated testing
- [ ] 2A - Sufficient functional/integration/UX testing exist
- [ ] 2B - Automated Test coverage
- [ ] 2C - The MockServer is used

#### 3. Code Health checks
- [ ] 3A - Code complexity is continuously low and decreasing over time
- [ ] 3B - Automated static code tests for security issues

#### 4. Automated Performance testing
- [ ] 4A - Code profiling (and query profiling) is part of the development process
- [ ] 4B - Response time is within defined SLAs
- [ ] 4C - The team understands proper sizing of the environment
- [ ] 4D - load and stress testing part of the CI pipeline and regressions are fixed resp. blocked for deployment
- If mobile, measures: 
    - [ ] 4E.i - FPS in % ()
    - [ ] 4E.ii - Consuming CPU in % (), memory in % (), diskspace in % (), battery in % ()
    - [ ] 4E.iii - Binary size in % ()
    - [ ] 4E.iii - Method number in Android in % () 
- [ ] 4F - Dynamic security checks
- [ ] 4G - Threat modeling
           
           
## Security

#### 1. Basics

- [ ] 1A - Centralized log
- [ ] 1B - Centralized authentication
- [ ] 1C - Centralized authorization
- [ ] 1D - External access via TLS 1.2
- [ ] 1E - Encrypted passwords using SHA-256 using SALT
- [ ] 1F - Passwords being trafficked over the networks must be encrypted
- [ ] 1G - Separation of duty and access
- [ ] 1H - Aware of SLA for security fixes

#### 2. Training & Internet Access Policies

- [ ] 2A.i   - security awareness
- [ ] 2A.ii  - secure development cycle
- [ ] 2A.iii - secure data handling
- [ ] 2D.iv  - secure OWASP

#### 3. Incident management, learning reviews, post mortems, etc.

- [ ] 3A - Incident SLAs are respected

#### 4. Secure Infrastructure

- [ ] 4A - Secure Infrastructure: hardening
- [ ] 4B - Secure Infrastructure: latest patches installed

#### 5. Continuous Security

- [ ] 5A - Automated pen tests for infrastructure
- [ ] 5A - Automated pen tests for application (OWSASP)
