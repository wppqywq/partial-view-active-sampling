# Publishing on GitHub

The workspace has a protected, empty `.git` placeholder managed by the application. It is not a Git repository. Do not delete or replace that directory. Export the selected source tree into a normal directory first.

```bash
bash scripts/check_release.sh
bash scripts/export_public.sh ../active-sensing-public
cd ../active-sensing-public
git init -b main
git add .
git diff --cached --stat
```

Before the first public push, select a project license that you have authority to grant and add its exact text as `LICENSE`. No license has been chosen on your behalf. Do not apply that license to external datasets, model weights, or third-party packages. The release includes aggregate plots rather than redistributed COCO photographs or participant-level gaze data.

Create an empty public repository on GitHub named `active-sensing` or another name you choose. Do not initialize it with a second README. Then, using your own configured Git identity:

```bash
git commit -m "Publish active sensing source and result snapshot"
git remote add origin https://github.com/YOUR_ACCOUNT/active-sensing.git
git push -u origin main
```

Replace `YOUR_ACCOUNT` and the repository name with the actual destination. No repository has been created or pushed by this cleanup.

The export uses an explicit file/directory allowlist and applies `.gitignore`, omitting application metadata, live job output, large experiment files, and private runtime data. `scripts/check_ignore.sh` checks both excluded runtime paths and retained public result files. Run `scripts/check_release.sh` again after adding or editing files. Review the staged diff before publishing; the automated check is a bounded content scan, not a security certification.
