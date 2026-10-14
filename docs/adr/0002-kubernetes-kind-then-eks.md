# ADR 0002: Develop on kind, release to EKS

Status: accepted

## Context

Kubernetes appears in roughly 29 percent of the AI engineer job descriptions sampled in a January 2026 analysis of 889 postings from European and American cities. A simpler managed container service would be enough for this application alone.

## Decision

Run every component as a Kubernetes workload from the start, develop on a local kind cluster, and deploy to EKS once for a short-lived demonstration, then destroy it. The EKS control plane is billed at 0.10 US dollars per hour, so the cluster is not kept running.

## Consequences

More configuration than a managed container service would need. In exchange, the project demonstrates probes, rolling updates, rollback, autoscaling and identity-based access to cloud services on a real cluster.
