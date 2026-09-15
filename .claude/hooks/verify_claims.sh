#!/bin/bash
# verify_claims.sh — Stop hook:逐條讀 .claude/claims.tsv,獨立重算真值與「宣稱值」比對。
# 不符 → 印出差異並 exit 2(擋住結束,強制修正)。由 harness 自動執行,不經模型自律。
# claims.tsv 格式(TAB 分隔):  key <TAB> 宣稱值 <TAB> 重算指令(於 repo 根執行,印出真值)
cd "$(dirname "$0")/../.." || exit 0
CLAIMS=".claude/claims.tsv"
[ -f "$CLAIMS" ] || exit 0
fail=0; out=""
while IFS=$'\t' read -r key claimed cmd; do
  [ -z "$key" ] && continue
  case "$key" in \#*) continue;; esac
  actual=$(bash -c "$cmd" 2>/dev/null | tr -d '[:space:]')
  claimed_t=$(printf '%s' "$claimed" | tr -d '[:space:]')
  if [ "$actual" != "$claimed_t" ]; then
    fail=1
    out="${out}
  ✗ ${key}: 宣稱=${claimed_t}  實算=${actual}   (重算: ${cmd})"
  fi
done < "$CLAIMS"
if [ "$fail" = 1 ]; then
  printf '【claims 驗證失敗】宣稱值與獨立重算不符,不得結束,請修正:%s\n' "$out" >&2
  exit 2
fi
exit 0
