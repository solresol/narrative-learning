-- View mapping round IDs to their investigation ID
-- Each *_rounds table stores the investigation that produced a round.
-- This view unions all of them so scripts can look up an investigation
-- given only a round_id.

CREATE OR REPLACE VIEW round_investigations AS
    SELECT round_id, investigation_id FROM espionage_rounds
  UNION ALL
    SELECT round_id, investigation_id FROM potions_rounds
  UNION ALL
    SELECT round_id, investigation_id FROM southgermancredit_rounds
  UNION ALL
    SELECT round_id, investigation_id FROM timetravel_insurance_rounds
  UNION ALL
    SELECT round_id, investigation_id FROM titanic_rounds
  UNION ALL
    SELECT round_id, investigation_id FROM wisconsin_rounds;

