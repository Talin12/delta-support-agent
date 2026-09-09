"""Hand-assigned golden labels, kept in source so every label is reviewable and diffable.

Each entry is (intent, route, route_reason). Labels were assigned by the author
against data/golden/codebook.md, in one pass plus a consistency re-read. Where the
re-read found inconsistency, the codebook rule was tightened and all affected cases
were re-decided together rather than patched individually; those three added rules
are recorded in the codebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prep import read_jsonl, write_jsonl  # noqa: E402

GOLDEN = REPO_ROOT / "data" / "golden"

A = "AUTO"
E = "ESCALATE"

LABELS: dict[str, tuple[str, str, str]] = {
    "g0001": ("praise", A, "Pure gratitude with no outstanding request."),
    "g0002": ("other", A, "Travel-moment chatter with no request."),
    "g0003": ("baggage_issue", A, "Routine bag-delay question; standard ask-for-details reply fits."),
    "g0004": ("other", A, "Excitement post, no request."),
    "g0005": ("service_complaint", A, "Sarcastic hold-time complaint; standard acknowledgment is low risk."),
    "g0006": ("service_complaint", E, "E2: alleges a staff member threatened the customer."),
    "g0007": ("seat_or_upgrade", A, "Routine question about comfort-seat eligibility."),
    "g0008": ("refund_or_compensation", E, "E5: duplicate charge requires a billing reversal."),
    "g0009": ("refund_or_compensation", E, "E5: refund sought for undelivered onboard service."),
    "g0010": ("praise", A, "Compliment comparing favourably to a competitor."),
    "g0011": ("praise", A, "Thanks naming an employee."),
    "g0012": ("refund_or_compensation", E, "E5: asks to reverse a SkyMiles purchase."),
    "g0013": ("flight_disruption", E, "E9: asks to hold a connecting flight in real time."),
    "g0014": ("service_complaint", A, "Sarcastic dissatisfaction with no specific claim."),
    "g0015": ("other", A, "Travel-moment chatter."),
    "g0016": ("flight_disruption", A, "Cancellation with a notification gap; standard reply."),
    "g0017": ("praise", A, "Brand compliment."),
    "g0018": ("flight_disruption", A, "Routine delay venting."),
    "g0019": ("policy_question", A, "Procedural request for a baggage receipt."),
    "g0020": ("service_complaint", E, "E1: customer states they will hire an attorney."),
    "g0021": ("flight_disruption", E, "E9: passengers stranded overnight awaiting refuelling."),
    "g0022": ("refund_or_compensation", E, "E5: explicitly asks what compensation will be paid."),
    "g0023": ("refund_or_compensation", E, "E5: chasing an open compensation case by number."),
    "g0024": ("flight_disruption", A, "Repeat-delay venting; routine."),
    "g0025": ("flight_disruption", A, "Snarky delay complaint; routine."),
    "g0026": ("praise", A, "Compliment about a Sky Club."),
    "g0027": ("policy_question", A, "Check-in procedure question with standard guidance."),
    "g0028": ("other", A, "Travel-moment chatter."),
    "g0029": ("other", A, "Too vague to act on, but a clarifying question is a safe reply."),
    "g0030": ("service_complaint", A, "Grievance about a carry-on fee; no refund demanded."),
    "g0031": ("policy_question", A, "Asks whether mid-flight seat changes are permitted."),
    "g0032": ("loyalty_account", A, "MQD qualification question with a documented answer."),
    "g0033": ("service_complaint", E, "E2/E9: requested wheelchair assistance denied, now missing a connection."),
    "g0034": ("flight_disruption", A, "Tarmac-delay venting."),
    "g0035": ("praise", A, "Compliment on crew and catering."),
    "g0036": ("booking_change", A, "Asks to confirm award ticketing status."),
    "g0037": ("other", A, "Product suggestion, not a support request."),
    "g0038": ("flight_disruption", A, "Waiting on a gate assignment."),
    "g0039": ("service_complaint", A, "Complaint about published hold times."),
    "g0040": ("other", A, "Travel-moment chatter."),
    "g0041": ("praise", A, "Praise for an accommodation; the safety rule misfires on 'injured'."),
    "g0042": ("service_complaint", A, "Sarcastic complaint about a counter wait."),
    "g0043": ("service_complaint", A, "Meal did not meet dietary needs; acknowledgment fits."),
    "g0044": ("flight_disruption", A, "Disputes the stated delay cause; routine."),
    "g0045": ("baggage_issue", E, "E5: chasing an unresolved damaged-baggage claim with a payout."),
    "g0046": ("service_complaint", E, "E4: explicitly asking to speak to a human."),
    "g0047": ("other", A, "School research question, not support."),
    "g0048": ("praise", A, "Compliment on an on-time flight."),
    "g0049": ("flight_disruption", A, "Terse delay complaint."),
    "g0050": ("other", A, "Aircraft enthusiasm post."),
    "g0051": ("other", A, "Route-announcement chatter."),
    "g0052": ("service_complaint", A, "Complaint about callback wait."),
    "g0053": ("booking_change", A, "Asks to verify an infant-in-arms entry."),
    "g0054": ("flight_disruption", A, "Multiple aircraft swaps; downgrade already resolved."),
    "g0055": ("refund_or_compensation", E, "E5: seeks a refund for a fare-class downgrade."),
    "g0056": ("flight_disruption", A, "Asks what the options are during a delay."),
    "g0057": ("flight_disruption", A, "Tarmac and gate-availability complaint."),
    "g0058": ("baggage_issue", A, "Item left onboard; standard lost-property process."),
    "g0059": ("service_complaint", E, "E4: complains specifically about being unable to reach a real person."),
    "g0060": ("baggage_issue", E, "E5: 15-hour bag delay with 'how are you going to make this right'."),
    "g0061": ("refund_or_compensation", E, "E5: explicit compensation demand after a 3-hour delay."),
    "g0062": ("other", A, "Photo share."),
    "g0063": ("service_complaint", A, "Unspecified grievance; asking what happened is a safe reply."),
    "g0064": ("service_complaint", A, "Status passenger unhappy with bag check and reseating."),
    "g0065": ("praise", A, "Fan message."),
    "g0066": ("seat_or_upgrade", A, "Asks why an open first-class seat was not offered."),
    "g0067": ("refund_or_compensation", E, "E5: states an expectation of compensation."),
    "g0068": ("praise", A, "Positive comparison after a flight."),
    "g0069": ("flight_disruption", A, "Questions a 2.5-hour delay in clear weather."),
    "g0070": ("refund_or_compensation", E, "E5: complaint that a refund was refused."),
    "g0071": ("loyalty_account", A, "Gold benefits not showing; account refresh."),
    "g0072": ("booking_change", A, "Award booking site failing; defection remark alone is not escalation."),
    "g0073": ("other", A, "Airport photo with a pet."),
    "g0074": ("policy_question", A, "Question about snack serving size."),
    "g0075": ("other", A, "Sponsorship thanks; the legal rule misfires on the first name 'Sue'."),
    "g0076": ("service_complaint", A, "Complaint about staffing at bag drop."),
    "g0077": ("service_complaint", A, "Feedback about the Wi-Fi partner."),
    "g0078": ("refund_or_compensation", E, "E5: duplicate charge for a ticket."),
    "g0079": ("service_complaint", E, "E4/E6: a promised DM follow-up never happened."),
    "g0080": ("flight_disruption", E, "E9: 'never making this flight' at a congested airport, in progress."),
    "g0081": ("booking_change", A, "Known traveller number not showing on a reservation."),
    "g0082": ("other", A, "Profanity with a link and no discernible request."),
    "g0083": ("flight_disruption", A, "Asks for real-time flight status for an airport pickup."),
    "g0084": ("refund_or_compensation", E, "E5: asks for money back over catering."),
    "g0085": ("service_complaint", A, "Broken headphone jack; standard in-flight guidance."),
    "g0086": ("other", A, "College research question about recruitment."),
    "g0087": ("baggage_issue", A, "Bag left behind; standard delayed-baggage report is the right first step."),
    "g0088": ("refund_or_compensation", E, "E5: downgrade with an unconfirmed refund due."),
    "g0089": ("refund_or_compensation", E, "E5: refund sought for a Sky Club membership after lounge denial."),
    "g0090": ("service_complaint", A, "Asks where to send a complaint; standard link."),
    "g0091": ("service_complaint", A, "Check-in queue complaint; defection remark is not escalation."),
    "g0092": ("flight_disruption", A, "Asks about LAX delays; answerable from public status."),
    "g0093": ("refund_or_compensation", E, "E5: disputed refund an agent had already promised."),
    "g0094": ("other", A, "Sports poll unrelated to Delta support."),
    "g0095": ("refund_or_compensation", E, "E5: thousands of dollars of lost items with no reimbursement path."),
    "g0096": ("other", A, "Location check-in post."),
    "g0097": ("service_complaint", A, "Complaint that a mobile boarding pass was not accepted."),
    "g0098": ("other", A, "Question about a homepage photo location."),
    "g0099": ("praise", A, "Shout-out to Sky Club staff."),
    "g0100": ("service_complaint", A, "Generic complaint about rude employees."),
    "g0101": ("service_complaint", A, "Wi-Fi pricing rant; the fraud rule misfires on rhetorical 'fraud'."),
    "g0102": ("refund_or_compensation", E, "E5: asks where to request a refund after delays."),
    "g0103": ("praise", A, "Compliment on bag-tracking notifications."),
    "g0104": ("flight_disruption", A, "Sarcastic delay complaint."),
    "g0105": ("service_complaint", A, "Complaint about queue-cutting at check-in."),
    "g0106": ("service_complaint", A, "Reports a site outage; the security rule misfires on 'hacked'."),
    "g0107": ("praise", A, "Compliment on kids' meal quality."),
    "g0108": ("service_complaint", A, "Vague negative sentiment; a generic offer to help is safe."),
    "g0109": ("booking_change", A, "Needs reservation help but unspecified; a clarifying question fits."),
    "g0110": ("praise", A, "Thanks for a first-class upgrade."),
    "g0111": ("other", A, "Excited travel post."),
    "g0112": ("refund_or_compensation", E, "E5: 3-day delay of an infant safety seat with compensation raised."),
    "g0113": ("service_complaint", A, "Feedback about in-flight entertainment audio."),
    "g0114": ("service_complaint", A, "Complaint about a partner car rental rate."),
    "g0115": ("flight_disruption", A, "Repeated delays; asks about alternatives."),
    "g0116": ("praise", A, "Delayed shout-out to a gate agent."),
    "g0117": ("seat_or_upgrade", A, "Seat map misrepresented a window seat."),
    "g0118": ("other", A, "Travel itinerary chatter."),
    "g0119": ("refund_or_compensation", E, "E5: refund for an unused paid upgrade after a reroute."),
    "g0120": ("service_complaint", A, "Reports the website being down."),
    "g0121": ("baggage_issue", A, "Delays plus a suspected lost bag; the bag drives the first action."),
    "g0122": ("service_complaint", E, "E2: a medical dietary requirement was treated as a preference."),
    "g0123": ("refund_or_compensation", E, "E5: asks to cancel and be refunded due to an extreme delay."),
    "g0124": ("loyalty_account", A, "Locked SkyMiles account; standard support-line referral."),
    "g0125": ("praise", A, "Compliment on a flight."),
    "g0126": ("praise", A, "Customer brought treats for the crew."),
    "g0127": ("service_complaint", A, "Dismissive agent reported; defection hashtags alone are not escalation."),
    "g0128": ("baggage_issue", A, "Chasing a baggage receipt for an expense report."),
    "g0129": ("refund_or_compensation", E, "E5: explicitly requests compensation for a broken seat."),
    "g0130": ("service_complaint", A, "Unanswered assistance button; the media rule misfires on 'press the button'."),
    "g0131": ("service_complaint", A, "Complaint about a 2-hour hold time."),
    "g0132": ("other", A, "Sky lounge chatter."),
    "g0133": ("policy_question", A, "Carry-on weight allowance with a documented answer."),
    "g0134": ("flight_disruption", A, "Missed a connection at a closed gate; standard policy explanation."),
    "g0135": ("booking_change", A, "Asks why basic economy bars changes."),
    "g0136": ("other", A, "Observational anecdote with no request."),
    "g0137": ("service_complaint", E, "E6/E7: public fraud accusation aimed at Delta's agents."),
    "g0138": ("flight_disruption", E, "E9: asks for a gate agent to be sent to a gate right now."),
    "g0139": ("baggage_issue", E, "E6/E9: stranded 14 hours with a lost bag containing a wallet."),
    "g0140": ("baggage_issue", A, "Asks to confirm bag routing before an international connection."),
    "g0141": ("flight_disruption", A, "Asks why a flight turned around midair."),
    "g0142": ("praise", A, "Shout-out naming an employee."),
    "g0143": ("service_complaint", A, "Turned away from a Sky Club; policy explanation fits."),
    "g0144": ("other", A, "Asks for an urgent DM but states no issue; a clarifying reply is safe."),
    "g0145": ("policy_question", A, "Overweight baggage fee with a documented answer."),
    "g0146": ("refund_or_compensation", E, "E5: duplicate charge for a flight."),
    "g0147": ("praise", A, "Compliment on the Sky Club."),
    "g0148": ("refund_or_compensation", E, "E5: full refund under the risk-free cancellation window."),
    "g0149": ("flight_disruption", E, "E6: three delays then cancellation forcing an overnight drive."),
    "g0150": ("flight_disruption", E, "E6: 38 hours awake across four delays; severe distress."),
    "g0151": ("service_complaint", E, "E2: a passenger and infant fell with no staff response."),
    "g0152": ("booking_change", E, "E2/E9: travel interruption due to a medical emergency."),
    "g0153": ("other", A, "Playful birthday request for the Porsche transfer."),
    "g0154": ("booking_change", A, "Known traveller number disappearing at check-in."),
    "g0155": ("refund_or_compensation", E, "E5: paid extra for a forward seat, moved back, no refund."),
    "g0156": ("refund_or_compensation", E, "E5: asks for Wi-Fi reimbursement."),
    "g0157": ("flight_disruption", E, "E6: group misconnection abroad with staff refusing to hold."),
    "g0158": ("refund_or_compensation", E, "E5/E6: refund refused; publicly accuses Delta of stealing."),
    "g0159": ("other", A, "Vague tag with a link and no stated request."),
    "g0160": ("baggage_issue", A, "Implied bag damage complaint; standard claim guidance."),
    "g0161": ("baggage_issue", A, "Damaged bag; pointing to the claim process commits nothing."),
    "g0162": ("refund_or_compensation", E, "E5: full refund requested due to visa issues."),
    "g0163": ("booking_change", A, "Website error when changing an award trip."),
    "g0164": ("refund_or_compensation", E, "E5: double charged for checked baggage."),
    "g0165": ("flight_disruption", A, "Long tarmac wait under an ATC directive; standard explanation."),
    "g0166": ("baggage_issue", E, "E3/E6: luggage described as lost or stolen after days of failure."),
    "g0167": ("praise", A, "Thanks for care shown to a family member in the military."),
    "g0168": ("service_complaint", A, "Medallion denied Sky Priority lane access."),
    "g0169": ("other", A, "Photo share."),
    "g0170": ("refund_or_compensation", E, "E5: demands money back for a 6-hour delay then cancellation."),
    "g0171": ("service_complaint", A, "Difficulty redeeming a gift card."),
    "g0172": ("praise", A, "Loyalty statement."),
    "g0173": ("service_complaint", A, "Diamond member reports unanswered DMs."),
    "g0174": ("other", A, "Mixed observation about airport construction."),
    "g0175": ("booking_change", A, "Asks whether a purchased upgrade survives a voluntary change."),
    "g0176": ("other", A, "Excited first-time flyer post."),
    "g0177": ("praise", A, "Compliment on a Delta One flight."),
    "g0178": ("service_complaint", E, "E9: hold time may cost a time-critical trip."),
    "g0179": ("other", A, "Group travel photo post."),
    "g0180": ("refund_or_compensation", E, "E5: prepaid Wi-Fi not honoured; explicitly owed a refund."),
    "g0181": ("refund_or_compensation", E, "E5: demands a Wi-Fi refund."),
    "g0182": ("refund_or_compensation", E, "E5: caps-lock compensation demand after another delay."),
    "g0183": ("policy_question", A, "Asks whether a guest pass covers family."),
    "g0184": ("service_complaint", A, "Delta Experiences bidding site not working."),
    "g0185": ("refund_or_compensation", E, "E5: card charged twice for one set of tickets."),
    "g0186": ("flight_disruption", E, "E9: asks to be rebooked while deplaning right now."),
    "g0187": ("booking_change", A, "Duplicate itinerary; standard guidance to contact the agency."),
    "g0188": ("refund_or_compensation", E, "E5: asks for goodwill miles because a bag was lost."),
    "g0189": ("service_complaint", A, "Questions a 2-hour hold time."),
    "g0190": ("praise", A, "Positive note about through-checked bags."),
    "g0191": ("policy_question", A, "Asks whether Twitter handles reservation questions."),
    "g0192": ("flight_disruption", E, "E6: repeated contact attempts with no assistance on an international trip."),
    "g0193": ("flight_disruption", A, "Delay then return to gate; frustrated but routine."),
    "g0194": ("service_complaint", A, "Awaiting a callback since Friday."),
    "g0195": ("praise", A, "Thanks for legroom."),
    "g0196": ("praise", A, "Compliment naming a crew member."),
    "g0197": ("service_complaint", A, "Hung up after 25 minutes on the SkyMiles line."),
    "g0198": ("praise", A, "Compliment on in-flight Wi-Fi."),
    "g0199": ("flight_disruption", A, "Asks why a specific flight is delayed."),
    "g0200": ("refund_or_compensation", E, "E5: cancelled weeks ago and refund not received."),
    "g0201": ("praise", A, "Employee returned a lost iPhone."),
    "g0202": ("service_complaint", A, "Generalised complaint about repeated poor flights."),
    "g0203": ("praise", A, "Positive post about a flight."),
    "g0204": ("flight_disruption", A, "Family member in an 8-hour gate delay."),
    "g0205": ("seat_or_upgrade", A, "Silver member denied a move to an empty premium seat."),
    "g0206": ("other", A, "Excited travel post."),
    "g0207": ("refund_or_compensation", E, "E5/E7: refund dispute amplified by a podcast reaching millions."),
    "g0208": ("refund_or_compensation", E, "E5: refund tied to a hurricane cruise cancellation."),
    "g0209": ("service_complaint", A, "Complaint about rude staff with no specifics."),
    "g0210": ("other", A, "Applause emoji with a link."),
    "g0211": ("flight_disruption", A, "Four departures to get home from Hawaii."),
    "g0212": ("baggage_issue", A, "Damaged luggage; standard claim-filing guidance."),
    "g0213": ("service_complaint", E, "E2: alleges the aircraft should not have flown; an airworthiness claim."),
    "g0214": ("service_complaint", A, "Feedback about airport crowding design."),
    "g0215": ("seat_or_upgrade", A, "Medallion complains upgrades are never available."),
    "g0216": ("other", A, "Excited pre-trip post."),
    "g0217": ("service_complaint", E, "E6: all-caps public warning telling others not to buy Delta gift cards."),
    "g0218": ("other", E, "E7: asked to comment publicly on a contested matter."),
    "g0219": ("loyalty_account", A, "Asks why miles credit is missing for a ticket."),
    "g0220": ("flight_disruption", A, "Early flight delayed; frustrated but routine."),
}


def main() -> None:
    rows = read_jsonl(GOLDEN / "to_label.jsonl")
    missing = [r["golden_id"] for r in rows if r["golden_id"] not in LABELS]
    if missing:
        raise SystemExit(f"{len(missing)} unlabelled: {missing[:10]}")

    valid_intents = set()
    import json

    taxonomy = json.loads((GOLDEN / "taxonomy.json").read_text())
    valid_intents = {i["name"] for i in taxonomy["intents"]}

    out = []
    for row in rows:
        intent, route, reason = LABELS[row["golden_id"]]
        if intent not in valid_intents:
            raise SystemExit(f"{row['golden_id']}: unknown intent {intent!r}")
        out.append({**row, "intent": intent, "route": route, "route_reason": reason})

    write_jsonl(out, GOLDEN / "golden_set.jsonl")

    from collections import Counter

    print(f"labelled {len(out)} examples -> {GOLDEN / 'golden_set.jsonl'}\n")
    print("intent distribution:")
    for name, n in Counter(r["intent"] for r in out).most_common():
        print(f"  {name:26s} {n:4d}  ({100*n/len(out):.1f}%)")
    print("\nroute distribution:")
    for stratum in ("representative", "enriched"):
        sub = [r for r in out if r["stratum"] == stratum]
        esc = sum(1 for r in sub if r["route"] == "ESCALATE")
        print(f"  {stratum:16s} n={len(sub):3d}  escalate={esc:3d} ({100*esc/len(sub):.1f}%)")
    esc = sum(1 for r in out if r["route"] == "ESCALATE")
    print(f"  {'pooled':16s} n={len(out):3d}  escalate={esc:3d} ({100*esc/len(out):.1f}%)")


if __name__ == "__main__":
    main()
