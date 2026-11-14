---
doc_id: eng-oncall-runbook
title: On-Call Runbook
category: Engineering
owner: Platform Team
effective_date: 2025-06-15
status: current
---

# On-Call Runbook

## 1. Purpose and scope
This runbook describes how on-call works for the production services run by the Platform, Data and Robotics Firmware teams at Northgate Robotics. It covers rotations, paging, severity levels, escalation, communication during incidents, and what happens afterwards. Everyone who is on a rotation must read it before their first shift, and team leads re-read it after every post-incident review that changes it.

## 2. Rotations
Each team runs a weekly rotation with a primary and a secondary engineer. Rotations hand over on Mondays at 10:00 Central Time in the team's stand-up. The primary is paged first for every alert routed to the team; the secondary is the backstop. Engineers join the rotation after three months on the team and after shadowing two shifts. Swaps are arranged directly between engineers and recorded in the paging tool so that the schedule stays accurate. An on-call stipend of $500 per week is paid for the primary role and half that for the secondary role.

## 3. Being reachable
While on call, keep your phone charged and on loud, keep the paging app installed and logged in, and stay within reach of a laptop with VPN access. If you cannot be reachable for a stretch of more than an hour, arrange cover with the secondary and record it in the paging tool. Travel during an on-call week is discouraged; if it is unavoidable, swap the shift.

## 4. Severity levels
- **SEV1**: customer-facing outage or data loss affecting many customers, or any safety-related fault in deployed robots. Acknowledge within 15 minutes.
- **SEV2**: major degradation, a single large customer down, or a safety-adjacent fault with a workaround. Acknowledge within 45 minutes.
- **SEV3**: minor degradation or a problem with a clear workaround. Acknowledge within four hours.
- **SEV4**: cosmetic or low-impact issues handled in business hours.
The SEV Definitions document has the full matrix with examples.

## 5. Receiving a page
When you are paged, acknowledge in the paging app first, then look. Acknowledging stops the escalation timer and tells everyone that a human is on it. Open the alert's runbook link if there is one. Check the service dashboard, recent deploys in #deploys, and the status of upstream dependencies before forming a theory. Write what you see in the incident channel as you go; the notes are the raw material for the post-incident review and they help whoever joins you.

## 6. Escalation
If the primary on-call does not acknowledge a page within **15 minutes**, the paging tool pages the secondary automatically. If the secondary also fails to acknowledge within another 15 minutes, the page goes to the team's engineering manager. **An engineer who needs help on an active SEV1 should not wait for the timers: call the secondary directly through the incident phone bridge on extension 304**, which rings the secondary's phone and the manager's phone at the same time. Never escalate by Slack message alone during a SEV1; Slack is not a paging channel. For incidents that touch customer data or robot safety, also page the Security Lead and the Robotics Firmware on-call through the paging tool's "additional responders" option.

## 7. Declaring an incident
A SEV1 or SEV2 is declared by the responder, not by the alert. To declare, run the `/incident declare` command in Slack, which creates the incident channel, opens the status page draft, and posts a summary to **#sev1-bridge**, the company-wide channel for live SEV1 coordination. The person who declares becomes the incident commander until they hand over explicitly. The incident commander coordinates, keeps the timeline, and decides when to involve customer communications; they should not also be the person debugging if more than one responder is available.

## 8. Communication during an incident
Update the incident channel at least every 30 minutes for a SEV1 and every hour for a SEV2, even if the update is "no change". Customer-facing updates go on the status page through the Customer Support Escalation process; engineers do not post to customers directly. Keep speculation out of customer updates. Internally, say what you know, what you are doing, and when the next update will come.

## 9. Mitigation first
The goal during the incident is to restore service, not to find the root cause. Rolling back the last deploy is the default first action when an incident starts shortly after a deploy; see the Production Deploy Runbook for the rollback window. Feature flags are the second lever. Scaling up is the third. Record every action and its time in the channel.

## 10. Handover
If an incident runs longer than four hours, the incident commander arranges a handover to a fresh responder. Hand over by writing a summary in the channel covering impact, current theory, actions taken, and open questions, then confirm verbally that the new commander has read it.

## 11. Closing and post-incident review
Close the incident when customer impact has ended and the service is stable for 30 minutes. Every SEV1 and SEV2 gets a post-incident review within five working days, written by the incident commander with the responders. Reviews are blameless: they examine systems and decisions, not people. Action items get owners and dates and are tracked in the engineering backlog.

## 12. After your shift
At handover on Monday, summarise the week's pages in the stand-up, flag noisy alerts for tuning, and update this runbook if anything in it was wrong or missing.
