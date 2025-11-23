#!/bin/bash
# ChatGPT Git Agent - safer version with ENDPATCH delimiter

set -e

REPO_DIR="$(pwd)"

echo "[Agent] Working directory: $REPO_DIR"
echo "[Agent] Waiting for ChatGPT instructions…"
echo

while IFS= read -r line; do
    case "$line" in

        CHECKOUT*)
            BRANCH="${line#CHECKOUT }"
            git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
            echo "[Agent] Switched to branch: $BRANCH"
            ;;

        APPLY_PATCH*)
            PATCH_FILE="/tmp/chatgpt_patch_$$.diff"
            echo "[Agent] Reading patch until ENDPATCH..."
            : > "$PATCH_FILE"  # truncate/clear file
            while IFS= read -r patch_line; do
                if [ "$patch_line" = "ENDPATCH" ]; then
                    break
                fi
                echo "$patch_line" >> "$PATCH_FILE"
            done
            echo "[Agent] Applying patch..."
            git apply "$PATCH_FILE"
            echo "[Agent] Patch applied."
            rm "$PATCH_FILE"
            ;;

        COMMIT*)
            MSG="${line#COMMIT }"
            git add -A
            git commit -m "$MSG"
            echo "[Agent] Committed: $MSG"
            ;;

        PUSH*)
            BRANCH="${line#PUSH }"
            git push -u origin "$BRANCH"
            echo "[Agent] Branch pushed: $BRANCH"
            ;;

        *)
            echo "[Agent] Unknown command: $line"
            ;;
    esac
done
