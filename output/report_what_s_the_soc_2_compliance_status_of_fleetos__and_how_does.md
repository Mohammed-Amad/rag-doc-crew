# What's the SOC 2 compliance status of FleetOS, and how does that relate to the data classification rules in the employee handbook?
## Executive Summary
FleetOS has completed a SOC 2 Type II audit with minor findings that were remediated, and it has specific data encryption and retention policies in place. Key points about its compliance status and data handling include:
* FleetOS completed a **SOC 2 Type II audit** in the prior fiscal year with **two minor findings**, both remediated within **30 days**.
* Customer facility layout and telemetry data are encrypted at rest (**AES-256**) and in transit (**TLS 1.2+**).
* Data retention for robot telemetry defaults to **18 months** unless a customer requests a shorter window.
* Warehouse camera and LIDAR data are classified as **Restricted** by default, with specific handling rules.
* **Restricted data** may not be downloaded to personal devices or personal cloud storage, and any approved export must record the ticket number, purpose, and retention period.
* **Confidential data** requires need-to-know access, and **Public data** may be shared freely.

## Notes
The information provided reflects the SOC 2 compliance status and data classification rules as described in the product specification and employee handbook documents. The relation between SOC 2 compliance and the data classification rules in the employee handbook is not addressed in the available documents.

## Flagged / Unverifiable
The relation between SOC 2 compliance and data classification rules in the employee handbook is not available in the provided documents.

## Citations
1. The SOC 2 compliance status of FleetOS is that it completed a SOC 2 Type II audit in the prior fiscal year with two minor findings, both remediated within 30 days: (product_spec.pdf, chunk 10)
2. Customer facility layout and telemetry data are encrypted at rest (AES-256) and in transit (TLS 1.2+): (product_spec.pdf, chunk 10)
3. Data retention for robot telemetry defaults to 18 months unless a customer requests a shorter window: (product_spec.pdf, chunk 10)
4. Warehouse camera and LIDAR data captured by deployed AMR fleets may contain people, vehicles, or facility layouts and is classified Restricted by default: (employee_handbook.pdf, chunk 16)
5. Restricted data may not be downloaded to personal devices or personal cloud storage, and any approved export must record the ticket number, purpose, and retention period: (employee_handbook.pdf, chunk 16)
6. Confidential data (contracts, employee records, unpublished financials) requires need-to-know access: (employee_handbook.pdf, chunk 16)
7. Public data (approved marketing material) may be shared freely: (employee_handbook.pdf, chunk 16)