param(
  [Parameter(Mandatory=$true)] [string]$InPath,
  [string]$OutPath = ""
)

function Num([string]$x) { if ($x -match '[-+]?\d+(\.\d+)?') { [double]$matches[0] } else { $null } }

# Parse teams from filename like NCAAF_BAYLOR@OKLAST_Projections_*.csv
$fname = Split-Path $InPath -Leaf
$tm = [regex]::Match($fname, "NCAAF_([^@]+)@([^_]+)_", 'IgnoreCase')
$teamFromName = $tm.Groups[1].Value
$oppFromName  = $tm.Groups[2].Value

# Where to write
if (-not $OutPath -or [IO.Directory]::Exists($OutPath)) {
  $base = [IO.Path]::GetFileNameWithoutExtension($fname)
  $dir  = ($OutPath -and (Test-Path $OutPath -PathType Container)) ? $OutPath : "C:\CFB_Engine\slates\cfb"
  $OutPath = Join-Path $dir ("converted_{0}.csv" -f $base)
}

$rows = Import-Csv -Path $InPath
$result = @()

foreach ($r in $rows) {
  # Player column finder
  $player = $r.Player
  if (-not $player) { $player = $r.Name }
  if (-not $player) { continue }

  # Collect props from common SportsLine headers (add more if needed)
  $cands = @(
    @{ PT="Passing Yards";      V=$r.PASSYD },
    @{ PT="Rushing Yards";      V=$r.RUSHYD },
    @{ PT="Receiving Yards";    V=$r.RECYD  },
    @{ PT="Rush + Rec TDs";     V=$r.RUSHTD },
    @{ PT="Rush + Rec TDs";     V=$r.RECTD  },
    @{ PT="Passing Touchdowns"; V=$r.TD     },
    @{ PT="Interceptions";      V=$r.INTS   }
  )

  foreach ($c in $cands) {
    $val = Num ($c.V)
    if ($null -ne $val) {
      $team = if ($r.Team) { $r.Team } else { $teamFromName }
      $opp  = if ($r.Opponent) { $r.Opponent } elseif ($r.Opp) { $r.Opp } else { $oppFromName }

      $result += [pscustomobject]@{
        Player    = $player
        PropType  = $c.PT
        Line      = ""
        Projection= $val
        Odds      = -110
        MyProb    = ""
        League    = "CFB"
        Team      = $team
        Opponent  = $opp
        Sigma     = ""
      }
    }
  }
}

if ($result.Count -eq 0) {
  Write-Host "No numeric projections found in $InPath" -ForegroundColor Yellow
} else {
  $result | Export-Csv -NoTypeInformation -Encoding UTF8 $OutPath
  Write-Host "Wrote $($result.Count) rows → $OutPath" -ForegroundColor Green
}
