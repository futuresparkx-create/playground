// monetization/vscode_extension/extension.js
// VS Code Extension Skeleton

// monetization/vscode_extension/
//    ├── package.json
//    ├── extension.js
//    └── src/
//         └── backend_api_stub.py

const vscode = require('vscode');

function activate(context) {
    let disposable = vscode.commands.registerCommand(
        'codemodel.fixCode',
        function () {
            vscode.window.showInformationMessage('Fixing code… (stub)');
        }
    );
    context.subscriptions.push(disposable);
}

module.exports = { activate };