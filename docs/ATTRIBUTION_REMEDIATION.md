# CC BY-NC 4.0 attribution and release remediation

This runbook is a maintainer workflow, not legal advice. Ask counsel to review the
response when a release may have been widely redistributed or termination rules
may apply.

## Licensor request to remove attribution information

CC BY-NC 4.0 Section 3(a)(3) allows the Licensor to request removal of the
Section 3(a)(1)(A) information to the extent reasonably practicable. Normal
release immutability is not an absolute bar to that compliance action.

1. Pause new publication for the affected model family and preserve the request
   and relevant release records for the compliance log.
2. Update the registry and preservation/rendering policy to reflect the request;
   do not infer or add an individual creator not designated by the Licensor.
3. Identify affected tags and downloadable assets with the read-only auditor:
   `python -m scripts.audit_cc_by_nc_releases --catalog catalog/bundles.json`
   Add local unpacked candidate directories to inspect the files and ZIP contents.
4. Remove or replace affected downloadable artifacts where reasonably
   practicable, coordinating any GitHub release/tag changes with maintainers.
   The duplicate-tag publication guard remains the default; an authorized legal
   remediation may use the documented manual exception process.
5. Publish corrected artifacts under an appropriately reviewed release identity,
   rebuild and validate the catalog, and stop advertising superseded assets.
6. Record the request, decision, affected tags, actions, corrected artifact
   identifiers, and catalog update in the compliance action log.

Do not rewrite release assets automatically. A maintainer must review and approve
any change to published releases or catalog records.

## Audit and remediate existing releases

1. Enumerate `phonon-40m-*` rows in `catalog/bundles.json`; the auditor reports
   entries with missing license/source/attribution metadata as requiring review.
2. Download each affected release ZIP and manifest to a controlled workspace,
   unpack the ZIP, and pass the unpacked candidate directory to
   `python -m scripts.audit_cc_by_nc_releases --catalog catalog/bundles.json PATH...`.
   The script is read-only; it does not fetch or alter remote releases.
   Keep each original `<tag>.zip` beside its extracted `<tag>/` directory so the auditor can compare the archive contents.
3. If attribution or required metadata is absent, stop advertising that asset in
   the bundle catalog until corrected. Inspect the upstream revision and retain
   the source model card and notices when preparing the corrected bundle.
4. Publish a corrected release or remove/replace the affected assets as
   appropriate, refresh `catalog/bundles.json`, and record the correction.
5. For artifacts already distributed broadly, promptly assess the defect and
   remediation with counsel; do not treat the acknowledgement flag as a cure for
   a license-condition failure.
