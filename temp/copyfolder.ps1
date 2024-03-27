# tsv파일에 있는 폴더명 list를 가져와서 그 폴더들만 다른 폴더로 복사

# cmd에서 network drive 사용 : net use \\163.180.160.71\DATA001 후에 \\163.180.160.71\DATA001 로 사용
# PowerShell을 관리자 모드로 실행하고, 스크립트 실행 정책을 변경: Set-ExecutionPolicy RemoteSigned
# 이 파일 위치로 가서 cd C:\Users\JihooPark\MyProjects\GNN_SCHZ\temp
# .\copyfolder.ps1

$tsvFile = "\\163.180.160.71\DATA001\MRI\DecNef\origin\SRPBS_1600\participants.tsv" # TSV 파일 경로
$sourceDir = "\\163.180.160.71\DATA001\MRI\DecNef\processed\freesurfer" # 원본 폴더가 있는 경로
$destDir = "\\163.180.160.71\DATA001\MRI\DecNef\unprocess\1600\BIDS_format\derivatives\sourcedata\freesurfer" # 폴더를 복사할 대상 경로

$data = Import-Csv -Path $tsvFile -Delimiter "`t"

foreach ($row in $data) {
    if ($row.diag -eq "0" -or $row.diag -eq "4") { # SCZ, HC만 선택
        $id = $row.participant_id -replace '[^\d]', ''
        $folderName = "${id}_mprage" # 폴더명 sub-에서 mprage로 변경
        $destFolderName = "sub-${id}" #복사한 폴더명은 다시 sub-형태로 변경

        $sourcePath = Join-Path -Path $sourceDir -ChildPath $folderName
        $destPath = Join-Path -Path $destDir -ChildPath $destFolderName
        
        # 폴더 복사
        if (Test-Path -Path $sourcePath) {
            Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
        } else {
            Write-Host "Folder not found: $sourcePath"
        }
    }
}