:: Deploy on GitHub pages.
:: See https://cli.vuejs.org/guide/deployment.html#github-pages

call yarn build

pushd dist

:: The models are too large, delete them to avoid pushing to GitHub.
for /d /r %%i in (*.tfjs) do @rmdir /S /Q "%%i"

git init
git add -A
git commit -m 'deploy'

git push -f git@github.com:ivan-alles/preference-model.git master:gh-pages

popd
