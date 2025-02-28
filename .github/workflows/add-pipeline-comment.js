const issue = context.payload.issue;
const body = issue.body;

const fields = {
  'Pipeline Name': 'name',
  'GitHub URL': 'github_url',
  'License': 'license',
  'Primary Language': 'primary_language',
  'Pipeline Description': 'pipeline_description',
  'Demo (if available)': 'demo_link',
  'Has the pipeline been benchmarked?': 'benchmarked',
  'If yes, provide benchmark results or a link to evaluation metrics.': 'benchmark_results',
  'Additional Notes': 'additional_notes'
};

const extractedData = {};

for (const [label, id] of Object.entries(fields)) {
  const regex = new RegExp(`### ${label}\\n\\n(.*?)\\n\\n`, 's');
  const match = body.match(regex);
  extractedData[id] = match ? match[1].trim() : '';
}

const { name, github_url, license, primary_language, pipeline_description, demo_link, benchmarked, benchmark_results, additional_notes } = extractedData;

const formattedOutput = `### ${name}
[![GitHub last commit](https://img.shields.io/github/last-commit/${github_url.replace('https://github.com/', '')}?label=GitHub&logo=github)](${github_url})
![GitHub License](https://img.shields.io/github/license/${github_url.replace('https://github.com/', '')})
${demo_link ? `[![Demo](https://img.shields.io/badge/DEMO-black?logo=awwwards)](${demo_link})` : ''}
<!--- 
License: ${license}
Primary language: ${primary_language}
-->

${name} *“${pipeline_description}”*.

${demo_link ? `**Demo available at ${demo_link}**\n\n` : ''}${additional_notes ? `**Additional Notes:** ${additional_notes}\n\n` : ''}`;

console.log(formattedOutput);

const { owner, repo, number } = context.issue;

github.rest.issues.createComment({
  owner,
  repo,
  issue_number: number,
  body: formattedOutput
});
