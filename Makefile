lint:
	poetry run isort src
	poetry run black -l 79 src
	poetry run flake8 src

## 実行
repro: check_commit PIPELINE.md
	poetry run dvc repro
	git commit dvc.lock -m 'dvc repro を実行したため' || true

## パイプライン出力
PIPELINE.md: params.yaml dvc.yaml
	echo '# pipeline DAG' > $@
	poetry run dvc dag --md >> $@
	git commit $@ -m 'DVC パイプラインを変更したため' || true

## コミット漏れのチェック
check_commit:
	git status
	git diff --exit-code
	git diff --exit-code --staged
